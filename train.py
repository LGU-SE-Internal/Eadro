import logging
from pathlib import Path
from typing import List, Optional
import torch
import dgl
import typer
from torch.utils.data import Dataset, DataLoader

from src.eadro.utils import (
    load_chunks,
    read_json,
    dump_params,
    dump_scores,
    seed_everything,
)
from src.eadro.base import BaseModel
from src.eadro.config import Config


class ChunkDataset(Dataset):
    def __init__(self, chunks: dict, node_num: int, edges: list):
        self.data = []
        self.idx2id = {}

        for idx, chunk_id in enumerate(chunks.keys()):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]

            try:
                graph = dgl.graph(edges, num_nodes=node_num)
            except Exception:
                graph = dgl.DGLGraph()
                graph.add_nodes(node_num)
                if edges:
                    graph.add_edges(edges[0], edges[1])

            graph.ndata["logs"] = torch.FloatTensor(chunk["logs"])
            graph.ndata["metrics"] = torch.FloatTensor(chunk["metrics"])
            graph.ndata["traces"] = torch.FloatTensor(chunk["traces"])

            self.data.append((graph, chunk["culprit"]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def get_chunk_id(self, idx: int) -> str:
        return self.idx2id[idx]


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")


def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def setup_logging(log_file: Optional[str] = None):
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=handlers,
    )


def load_data(config: Config) -> tuple:
    chunks_dir = config.get("chunks_dir")
    if chunks_dir is None:
        raise ValueError("chunks_dir is not configured")

    data_dir = Path(chunks_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(str(metadata_path))
        if metadata is None:
            raise ValueError("Failed to read metadata file")
    else:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    config.set("event_num", metadata["event_num"])
    config.set("node_num", metadata["node_num"])
    config.set("metric_num", metadata["metric_num"])
    config.set("chunk_length", metadata["chunk_length"])

    train_chunks, test_chunks = load_chunks(str(data_dir))

    edges = metadata.get("edges", [])

    return train_chunks, test_chunks, edges, metadata


def create_data_loaders(
    train_chunks: dict, test_chunks: dict, node_num: int, edges: list, config: Config
) -> tuple:
    train_dataset = ChunkDataset(train_chunks, node_num, edges)
    test_dataset = ChunkDataset(test_chunks, node_num, edges)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size"),
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size"),
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_model(config: Config, evaluation_epoch: int = 10) -> tuple:
    random_seed = config.get("random_seed")
    if random_seed is None:
        raise ValueError("random_seed is not configured")
    seed_everything(random_seed)

    gpu_config = config.get("gpu")
    if gpu_config is None:
        raise ValueError("gpu setting is not configured")
    device = get_device(gpu_config)

    train_chunks, test_chunks, edges, metadata = load_data(config)

    node_num = config.get("node_num")
    if node_num is None:
        raise ValueError("node_num is not configured")

    train_loader, test_loader = create_data_loaders(
        train_chunks, test_chunks, node_num, edges, config
    )

    model = BaseModel(
        device=str(device),
        **config.to_dict(),
    )

    scores, converge = model.fit(
        train_loader, test_loader, evaluation_epoch=evaluation_epoch
    )

    return scores, converge


app = typer.Typer()


@app.command()
def main(
    random_seed: int = typer.Option(42, help="Random seed"),
    gpu: bool = typer.Option(True, help="Use GPU"),
    epochs: int = typer.Option(500, help="Training epochs"),
    batch_size: int = typer.Option(256, help="Batch size"),
    lr: float = typer.Option(0.01, help="Learning rate"),
    patience: int = typer.Option(10, help="Early stopping patience"),
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
    chunks_dir: str = typer.Option("dataset_output", help="Chunks directory"),
    dataset: str = typer.Option("rcabench", help="Dataset name"),
    use_wandb: bool = typer.Option(True, help="Use Weights & Biases for logging"),
    wandb_project: str = typer.Option("eadro-training", help="W&B project name"),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
):
    config_dict = {
        "random_seed": random_seed,
        "gpu": gpu,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
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
        "chunks_dir": chunks_dir + "/" + dataset,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
    }

    if config_file and Path(config_file).exists():
        config = Config(config_file)
        for key, value in config_dict.items():
            config.set(key, value)
    else:
        config = Config()
        for key, value in config_dict.items():
            config.set(key, value)

    result_dir_config = config.get("result_dir")
    if result_dir_config is None:
        raise ValueError("result_dir is not configured")
    result_dir_path = Path(result_dir_config)
    result_dir_path.mkdir(parents=True, exist_ok=True)

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
