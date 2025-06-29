from pathlib import Path
from typing import List, Optional, Tuple
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
from src.preprocessing.base import (
    DataSample,
    DatasetMetadata,
    TimeSeriesDatapack,
)


class WindowDataset(Dataset):
    """Dataset for pre-computed window data samples"""

    def __init__(
        self,
        window_data: List[Tuple[DataSample, int]],
        metadata: DatasetMetadata,
    ):
        self.metadata = metadata
        self.node_num = len(metadata.services)

        edges_src = []
        edges_dst = []
        for edge in metadata.service_calling_edges:
            edges_src.append(edge[0])
            edges_dst.append(edge[1])

        self.edges = (edges_src, edges_dst) if edges_src else ([], [])

        self.graphs = []
        self.labels = []

        logger.info("Pre-creating graphs...")
        for data_sample, label in window_data:
            graph = self._create_graph(data_sample)
            self.graphs.append(graph)
            self.labels.append(label)

        logger.info(
            f"Created WindowDataset with {len(self.graphs)} pre-computed graphs"
        )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, int]:
        return self.graphs[idx], self.labels[idx]

    def _create_graph(self, sample: DataSample) -> dgl.DGLGraph:
        assert len(self.edges) > 0, "Edges must be defined for the graph"
        graph = dgl.graph(self.edges, num_nodes=self.node_num)
        graph.ndata["logs"] = torch.FloatTensor(sample.log)
        graph.ndata["metrics"] = torch.FloatTensor(sample.metric)
        graph.ndata["traces"] = torch.FloatTensor(sample.trace)
        return graph


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


def load_timeseries_data(
    config: Config,
) -> Tuple[List[Tuple[DataSample, int]], List[Tuple[DataSample, int]], DatasetMetadata]:
    dataset_name = config.get("dataset")

    datapack_path = Path(f".cache/{dataset_name}_timeseries_datapack.pkl")
    assert datapack_path.exists()

    datapack = TimeSeriesDatapack.load(str(datapack_path))
    assert datapack is not None, "Time series datapack is None"
    assert datapack.metadata is not None, "Time series datapack metadata is None"

    window_size = config.get("datasets")[config.get("dataset")]["sample_interval"]
    step_size = config.get("datasets")[config.get("dataset")]["sample_step"]

    label_to_windows = defaultdict(list)

    for ts_sample in datapack.samples:
        total_time = (ts_sample.end_time - ts_sample.start_time).total_seconds()
        abnormal_time = 0
        main_gt_service = ""

        assert (
            len(ts_sample.abnormal_periods) == 1 or len(ts_sample.abnormal_periods) == 0
        )
        for start, end, gt_service, fault_type in ts_sample.abnormal_periods:
            abnormal_duration = (end - start).total_seconds()
            abnormal_time += abnormal_duration
            if abnormal_duration > 0:
                if not main_gt_service:
                    main_gt_service = gt_service

        abnormal_ratio = abnormal_time / total_time if total_time > 0 else 0
        if abnormal_ratio > 0.1:
            service_id = datapack.metadata.service_name_to_id.get(main_gt_service, -1)
        else:
            service_id = -1

        # Generate windows for this time series sample
        time_steps = ts_sample.get_time_steps()
        max_start = time_steps - window_size

        assert max_start >= 0
        for start_idx in range(0, max_start + 1, step_size):
            window_data = ts_sample.get_time_window(start_idx, window_size)
            label_to_windows[service_id].append((window_data, service_id))

    for label, windows in label_to_windows.items():
        logger.info(f"Service {label}: {len(windows)} windows")

    train_ratio = config.get("training.train_ratio")
    train_windows = []
    test_windows = []

    for label, windows in label_to_windows.items():
        label_windows = windows.copy()
        split_idx = int(len(label_windows) * train_ratio)
        train_windows.extend(label_windows[:split_idx])
        test_windows.extend(label_windows[split_idx:])

    logger.info(f"Total training windows: {len(train_windows)}")
    logger.info(f"Total testing windows: {len(test_windows)}")
    return train_windows, test_windows, datapack.metadata


def create_timeseries_data_loaders(
    train_windows: List[Tuple[DataSample, int]],
    test_windows: List[Tuple[DataSample, int]],
    metadata: DatasetMetadata,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    batch_size = config.get("training.batch_size")

    train_dataset = WindowDataset(
        window_data=train_windows,
        metadata=metadata,
    )

    test_dataset = WindowDataset(
        window_data=test_windows,
        metadata=metadata,
    )

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

        train_windows, test_windows, metadata = load_timeseries_data(config)
        train_loader, test_loader = create_timeseries_data_loaders(
            train_windows, test_windows, metadata, config
        )

        config.set("node_num", len(metadata.services))
        config.set("event_num", len(metadata.log_templates) + 1)
        config.set("metric_num", len(metadata.metrics))

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
