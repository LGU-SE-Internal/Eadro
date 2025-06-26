import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import torch
import dgl
import typer
from torch.utils.data import Dataset, DataLoader

from src.eadro.utils import (
    seed_everything,
    dump_params,
    dump_scores,
)
from src.eadro.base import BaseModel
from src.eadro.config import Config
from src.preprocessing.base import DataSample, DatasetMetadata


class ChunkDataset(Dataset):
    def __init__(self, samples: List[DataSample], metadata: DatasetMetadata):
        self.samples = samples
        self.metadata = metadata
        self.node_num = len(metadata.services)

        edges_src = []
        edges_dst = []
        for edge in metadata.service_calling_edges:
            edges_src.append(edge[0])
            edges_dst.append(edge[1])

        self.edges = (edges_src, edges_dst) if edges_src else ([], [])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, int]:
        sample = self.samples[idx]

        # Create DGL graph
        assert len(self.edges) > 0, "Edges must be defined for the graph"
        graph = dgl.graph(self.edges, num_nodes=self.node_num)

        # Add node features
        graph.ndata["logs"] = torch.FloatTensor(sample.log)
        graph.ndata["metrics"] = torch.FloatTensor(sample.metric)
        graph.ndata["traces"] = torch.FloatTensor(sample.trace)

        # Convert ground truth service to label
        if sample.gt_service and sample.gt_service in self.metadata.service_name_to_id:
            label = self.metadata.service_name_to_id[sample.gt_service]
        else:
            label = 0  # Default to first service if no ground truth

        return graph, label


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")


def collate_fn(
    batch: List[Tuple[dgl.DGLGraph, int]],
) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.long)


def setup_logging(log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=handlers,
    )


def load_data(
    config: Config,
) -> Tuple[List[DataSample], List[DataSample], DatasetMetadata]:
    """Load processed dataset samples and metadata"""
    dataset_name = config.get("dataset", "tt")

    # Load samples
    samples_path = Path(f".cache/{dataset_name}_samples.pkl")
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    with open(samples_path, "rb") as f:
        all_samples = pickle.load(f)

    # Load metadata
    metadata_path = Path(f".cache/{dataset_name}_metadata.pkl")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = DatasetMetadata.from_pkl(str(metadata_path))
    if metadata is None:
        raise ValueError(f"Failed to load metadata from {metadata_path}")

    # Split samples into train and test
    train_ratio = config.get("train_ratio", 0.7)
    if train_ratio is None:
        train_ratio = 0.7
    split_idx = int(len(all_samples) * train_ratio)

    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    logging.info(
        f"Loaded {len(train_samples)} training samples and {len(test_samples)} test samples"
    )
    logging.info(f"Number of services: {len(metadata.services)}")
    logging.info(f"Number of log templates: {len(metadata.log_templates)}")
    logging.info(f"Number of metrics: {len(metadata.metrics)}")
    logging.info(
        f"Number of service calling edges: {len(metadata.service_calling_edges)}"
    )

    return train_samples, test_samples, metadata


def create_data_loaders(
    train_samples: List[DataSample],
    test_samples: List[DataSample],
    metadata: DatasetMetadata,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and testing"""

    train_dataset = ChunkDataset(train_samples, metadata)
    test_dataset = ChunkDataset(test_samples, metadata)
    import numpy as np

    batch_size = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_model(
    config: Config, evaluation_epoch: int = 10
) -> Tuple[Dict[str, Any], bool]:
    random_seed = config.get("random_seed", 42)
    if random_seed is None:
        random_seed = 42
    seed_everything(random_seed)

    gpu_config = config.get("gpu", False)
    if gpu_config is None:
        gpu_config = False
    device = get_device(gpu_config)

    # Load data
    train_samples, test_samples, metadata = load_data(config)

    # Update config with dataset-specific parameters
    config.set("node_num", len(metadata.services))
    config.set("event_num", len(metadata.log_templates) + 1)  # +1 for unknown templates
    config.set("metric_num", len(metadata.metrics))

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_samples, test_samples, metadata, config
    )

    # Initialize model
    model = BaseModel(
        device=str(device),
        **config.to_dict(),
    )

    # Train model
    scores, converge = model.fit(
        train_loader, test_loader, evaluation_epoch=evaluation_epoch
    )

    # Handle None returns
    if scores is None:
        scores = {}
    if converge is None:
        converge = False
    else:
        converge = bool(converge)

    return scores, converge


app = typer.Typer()


@app.command()
def main(
    random_seed: int = typer.Option(42, help="Random seed"),
    gpu: bool = typer.Option(True, help="Use GPU"),
    epochs: int = typer.Option(50, help="Training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    lr: float = typer.Option(0.001, help="Learning rate"),
    patience: int = typer.Option(10, help="Early stopping patience"),
    train_ratio: float = typer.Option(0.7, help="Training data ratio"),
    lr_scheduler: str = typer.Option(
        "none", help="Learning rate scheduler: none, step, exponential, cosine, plateau"
    ),
    lr_step_size: int = typer.Option(50, help="Step size for StepLR scheduler"),
    lr_gamma: float = typer.Option(0.1, help="Multiplicative factor for LR decay"),
    lr_warmup_epochs: int = typer.Option(0, help="Number of warmup epochs"),
    lr_min: float = typer.Option(
        1e-5, help="Minimum learning rate for cosine scheduler"
    ),
    self_attn: bool = typer.Option(True, help="Use self attention"),
    fuse_dim: int = typer.Option(128, help="Fusion dimension"),
    alpha: float = typer.Option(0.5, help="Loss combination weight"),
    locate_hiddens: List[int] = typer.Option([64], help="Localization hidden dims"),
    detect_hiddens: List[int] = typer.Option([64], help="Detection hidden dims"),
    log_dim: int = typer.Option(16, help="Log embedding dimension"),
    trace_kernel_sizes: List[int] = typer.Option([2], help="Trace conv kernel sizes"),
    trace_hiddens: List[int] = typer.Option([64], help="Trace hidden dimensions"),
    metric_kernel_sizes: List[int] = typer.Option([2], help="Metric conv kernel sizes"),
    metric_hiddens: List[int] = typer.Option([64], help="Metric hidden dimensions"),
    graph_hiddens: List[int] = typer.Option([64], help="Graph hidden dimensions"),
    attn_head: int = typer.Option(4, help="Attention heads for GAT"),
    activation: float = typer.Option(0.2, help="LeakyReLU negative slope"),
    result_dir: str = typer.Option("result/", help="Result directory"),
    dataset: str = typer.Option("tt", help="Dataset name"),
    use_wandb: bool = typer.Option(True, help="Use Weights & Biases for logging"),
    wandb_project: str = typer.Option("eadro-training", help="W&B project name"),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
) -> None:
    """Train EADRO model on preprocessed dataset"""

    config_dict = {
        "random_seed": random_seed,
        "gpu": gpu,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "train_ratio": train_ratio,
        "lr_scheduler": lr_scheduler,
        "lr_step_size": lr_step_size,
        "lr_gamma": lr_gamma,
        "lr_warmup_epochs": lr_warmup_epochs,
        "lr_min": lr_min,
        "self_attn": self_attn,
        "fuse_dim": fuse_dim,
        "alpha": alpha,
        "locate_hiddens": locate_hiddens,
        "detect_hiddens": detect_hiddens,
        "log_dim": log_dim,
        "trace_kernel_sizes": trace_kernel_sizes,
        "trace_hiddens": trace_hiddens,
        "metric_kernel_sizes": metric_kernel_sizes,
        "metric_hiddens": metric_hiddens,
        "graph_hiddens": graph_hiddens,
        "attn_head": attn_head,
        "activation": activation,
        "result_dir": result_dir,
        "dataset": dataset,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
    }

    # Initialize config
    if config_file and Path(config_file).exists():
        config = Config(config_file)
        for key, value in config_dict.items():
            config.set(key, value)
    else:
        config = Config()
        for key, value in config_dict.items():
            config.set(key, value)

    # Setup result directory
    result_dir_config = config.get("result_dir")
    if result_dir_config is None:
        raise ValueError("result_dir is not configured")
    result_dir_path = Path(result_dir_config)
    result_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate hash ID and setup logging
    hash_id = dump_params(config.to_dict())
    log_file = result_dir_path / hash_id / "running.log"
    setup_logging(str(log_file))

    logging.info(f"Starting training with hash_id: {hash_id}")
    logging.info(f"Configuration: {config.to_dict()}")

    try:
        scores, converge = train_model(config)
        dump_scores(config.get("result_dir"), hash_id, scores, converge)
        logging.info(f"Training completed successfully. Hash ID: {hash_id}")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    app()
