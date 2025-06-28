import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import torch
import dgl
import typer
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import random
from collections import defaultdict
from src.eadro.utils import (
    seed_everything,
)
from src.eadro.base import BaseModel
from src.eadro.config import Config
from src.preprocessing.base import DataSample, DatasetMetadata
from typing import Counter


class ChunkDataset(Dataset):
    def __init__(
        self,
        samples: List[DataSample],
        metadata: DatasetMetadata,
        shuffle: bool = False,
    ):
        self.metadata = metadata
        self.node_num = len(metadata.services)

        # Shuffle samples at initialization if requested
        if shuffle:
            samples = samples.copy()
            random.shuffle(samples)

        self.samples = samples

        # Pre-build edges once
        edges_src = []
        edges_dst = []
        for edge in metadata.service_calling_edges:
            edges_src.append(edge[0])
            edges_dst.append(edge[1])

        self.edges = (edges_src, edges_dst) if edges_src else ([], [])

        # Pre-load all graphs to improve training speed
        logger.info(f"Pre-loading {len(samples)} graphs...")
        self.graphs = []
        self.labels = []

        for sample in self.samples:
            # Create DGL graph
            assert len(self.edges) > 0, "Edges must be defined for the graph"
            graph = dgl.graph(self.edges, num_nodes=self.node_num)

            # Add node features
            graph.ndata["logs"] = torch.FloatTensor(sample.log)
            graph.ndata["metrics"] = torch.FloatTensor(sample.metric)
            graph.ndata["traces"] = torch.FloatTensor(sample.trace)

            # Convert ground truth service to label
            label = sample.get_gt_service_id(self.metadata.service_name_to_id)

            self.graphs.append(graph)
            self.labels.append(label)

        # count the labels distribution
        label_counter = Counter(self.labels)
        logger.info("Label distribution:")
        for label, count in label_counter.items():
            logger.info(f"Service {label}: {count} samples")

        logger.info(f"Successfully pre-loaded {len(self.graphs)} graphs")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, int]:
        return self.graphs[idx], self.labels[idx]


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        logger.info("Using GPU...")
        return torch.device("cuda")
    logger.info("Using CPU...")
    return torch.device("cpu")


def collate_fn(
    batch: List[Tuple[dgl.DGLGraph, int]],
) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def setup_logging(log_file: Optional[str] = None) -> None:
    """Setup loguru logger with optional file output"""
    # Remove default handler first
    logger.remove()

    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} P{process} {level} {message}",
        level="INFO",
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} P{process} {level} {message}",
            level="INFO",
        )


def load_data(
    config: Config,
) -> Tuple[List[DataSample], List[DataSample], DatasetMetadata]:
    """Load processed dataset samples and metadata"""
    dataset_name = config.get("dataset")

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

    # Group samples by label (stratified sampling)
    label_to_samples = defaultdict(list)
    for sample in all_samples:
        label = sample.get_gt_service_id(metadata.service_name_to_id)
        label_to_samples[label].append(sample)

    # Log label distribution before balancing
    logger.info("Original label distribution:")
    for label, samples in label_to_samples.items():
        logger.info(f"Service {label}: {len(samples)} samples")

    # Balance the dataset by downsampling label -1 to match other classes
    if -1 in label_to_samples:
        # Find the minimum count among non-(-1) labels
        non_negative_counts = [
            len(samples) for label, samples in label_to_samples.items() if label != -1
        ]
        if non_negative_counts:
            min_count = min(non_negative_counts)

            # Downsample label -1 to match the minimum count
            negative_samples = label_to_samples[-1].copy()
            random.shuffle(negative_samples)
            label_to_samples[-1] = negative_samples[:min_count]

            logger.info(
                f"Downsampled label -1 from {len(negative_samples)} to {min_count} samples"
            )

    # Log balanced distribution
    logger.info("Balanced label distribution:")
    for label, samples in label_to_samples.items():
        logger.info(f"Service {label}: {len(samples)} samples")

    # Split each label's samples into train and test
    train_ratio = config.get("training.train_ratio")
    train_samples = []
    test_samples = []

    for label, samples in label_to_samples.items():
        # Shuffle samples for this label
        label_samples = samples.copy()
        random.shuffle(label_samples)

        # Split this label's samples
        split_idx = int(len(label_samples) * train_ratio)
        train_samples.extend(label_samples[:split_idx])
        test_samples.extend(label_samples[split_idx:])

    # Shuffle final train and test sets
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    logger.info(
        f"Loaded {len(train_samples)} training samples and {len(test_samples)} test samples"
    )

    # Log final distribution for verification
    train_label_counts = defaultdict(int)
    test_label_counts = defaultdict(int)

    for sample in train_samples:
        label = sample.get_gt_service_id(metadata.service_name_to_id)
        train_label_counts[label] += 1

    for sample in test_samples:
        label = sample.get_gt_service_id(metadata.service_name_to_id)
        test_label_counts[label] += 1

    logger.info("Training set label distribution:")
    for label, count in train_label_counts.items():
        logger.info(f"Service {label}: {count} samples")

    logger.info("Test set label distribution:")
    for label, count in test_label_counts.items():
        logger.info(f"Service {label}: {count} samples")

    logger.info(f"Number of services: {len(metadata.services)}")
    logger.info(f"Number of log templates: {len(metadata.log_templates)}")
    logger.info(f"Number of metrics: {len(metadata.metrics)}")
    logger.info(
        f"Number of service calling edges: {len(metadata.service_calling_edges)}"
    )

    return train_samples, test_samples, metadata


def create_data_loaders(
    train_samples: List[DataSample],
    test_samples: List[DataSample],
    metadata: DatasetMetadata,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    # Create datasets with shuffling for training data
    train_dataset = ChunkDataset(train_samples, metadata, shuffle=True)
    test_dataset = ChunkDataset(test_samples, metadata, shuffle=False)

    batch_size = config.get("training.batch_size")

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


app = typer.Typer()


@app.command()
def main(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    dataset: Optional[str] = typer.Option(None, help="Dataset name override"),
    epochs: Optional[int] = typer.Option(None, help="Training epochs override"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size override"),
    lr: Optional[float] = typer.Option(None, help="Learning rate override"),
    gpu: Optional[bool] = typer.Option(None, help="Use GPU override"),
    result_dir: Optional[str] = typer.Option(None, help="Result directory override"),
) -> None:
    config = Config(config_file)

    overrides = {
        "dataset": dataset,
        "training.epochs": epochs,
        "training.batch_size": batch_size,
        "training.lr": lr,
        "training.gpu": gpu,
        "paths.result_dir": result_dir,
    }

    for key, value in overrides.items():
        if value is not None:
            config.set(key, value)

    try:
        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)

        gpu_config = config.get("training.gpu")
        device = get_device(gpu_config)

        evaluation_epoch = config.get("training.evaluation_epoch")

        train_samples, test_samples, metadata = load_data(config)

        config.set("node_num", len(metadata.services))
        config.set("event_num", len(metadata.log_templates) + 1)
        config.set("metric_num", len(metadata.metrics))

        train_loader, test_loader = create_data_loaders(
            train_samples, test_samples, metadata, config
        )
        model = BaseModel(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=str(device),
            config=config,
        )

        model.fit(train_loader, test_loader, evaluation_epoch=evaluation_epoch)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    app()
