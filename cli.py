from pathlib import Path
from typing import Optional, List, Tuple
import torch
import dgl
import typer
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import random
from collections import defaultdict

from src.eadro.utils import seed_everything
from src.eadro.base import BaseModel
from src.eadro.config import Config
from src.preprocessing.base import (
    DataSample,
    DatasetMetadata,
    TimeSeriesDataSample,
    TimeSeriesDatapack,
)


class TimeWindowDataset(Dataset):
    def __init__(
        self,
        time_series_samples: List[TimeSeriesDataSample],
        metadata: DatasetMetadata,
        window_size: int,
        step_size: int,
        shuffle: bool = False,
    ):
        self.time_series_samples = time_series_samples
        self.metadata = metadata
        self.window_size = window_size
        self.step_size = step_size
        self.node_num = len(metadata.services)

        edges_src = []
        edges_dst = []
        for edge in metadata.service_calling_edges:
            edges_src.append(edge[0])
            edges_dst.append(edge[1])

        self.edges = (edges_src, edges_dst) if edges_src else ([], [])
        self.window_indices = self._compute_window_indices()

        if shuffle:
            random.shuffle(self.window_indices)

        logger.info(
            f"Created TimeWindowDataset with {len(self.window_indices)} windows from {len(time_series_samples)} time series samples"
        )

    def _compute_window_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for sample_idx, sample in enumerate(self.time_series_samples):
            time_steps = sample.get_time_steps()
            max_start = time_steps - self.window_size

            if max_start >= 0:
                for start_idx in range(0, max_start + 1, self.step_size):
                    indices.append((sample_idx, start_idx))
        return indices

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, int]:
        sample_idx, start_idx = self.window_indices[idx]
        time_series_sample = self.time_series_samples[sample_idx]
        data_sample = time_series_sample.get_time_window(start_idx, self.window_size)
        graph = self._create_graph(data_sample)
        label = data_sample.get_gt_service_id(self.metadata.service_name_to_id)
        return graph, label

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
) -> Tuple[List[TimeSeriesDataSample], List[TimeSeriesDataSample], DatasetMetadata]:
    dataset_name = config.get("dataset")
    datapack_path = Path(f".cache/{dataset_name}_timeseries_datapack.pkl")

    if not datapack_path.exists():
        raise FileNotFoundError(f"Time series datapack file not found: {datapack_path}")

    datapack = TimeSeriesDatapack.load(str(datapack_path))
    if datapack is None or datapack.metadata is None:
        raise ValueError(f"Failed to load time series datapack from {datapack_path}")

    label_to_samples = defaultdict(list)

    for ts_sample in datapack.samples:
        total_time = (ts_sample.end_time - ts_sample.start_time).total_seconds()
        abnormal_time = 0
        main_gt_service = ""

        for start, end, gt_service, fault_type in ts_sample.abnormal_periods:
            abnormal_duration = (end - start).total_seconds()
            abnormal_time += abnormal_duration
            if abnormal_duration > 0 and not main_gt_service:
                main_gt_service = gt_service

        abnormal_ratio = abnormal_time / total_time if total_time > 0 else 0
        service_id = (
            datapack.metadata.service_name_to_id.get(main_gt_service, -1)
            if abnormal_ratio > 0.1
            else -1
        )
        label_to_samples[service_id].append(ts_sample)

    # 打印初始 label 分布
    logger.info("=== 初始 Label 分布 ===")
    total_samples = sum(len(samples) for samples in label_to_samples.values())
    for service_id in sorted(label_to_samples.keys()):
        count = len(label_to_samples[service_id])
        percentage = count / total_samples * 100
        if service_id == -1:
            logger.info(
                f"Label {service_id} (正常样本): {count} 个样本 ({percentage:.1f}%)"
            )
        else:
            service_name = next(
                (
                    name
                    for name, id_ in datapack.metadata.service_name_to_id.items()
                    if id_ == service_id
                ),
                f"Unknown_{service_id}",
            )
            logger.info(
                f"Label {service_id} ({service_name}): {count} 个样本 ({percentage:.1f}%)"
            )

    # Balance dataset
    if -1 in label_to_samples:
        non_negative_counts = [
            len(samples) for label, samples in label_to_samples.items() if label != -1
        ]
        if non_negative_counts:
            min_count = min(non_negative_counts)
            negative_samples = label_to_samples[-1].copy()
            random.shuffle(negative_samples)
            original_negative_count = len(label_to_samples[-1])
            label_to_samples[-1] = negative_samples[:min_count]
            logger.info(
                f"数据集平衡: 正常样本从 {original_negative_count} 减少到 {min_count}"
            )

    # 打印平衡后的 label 分布
    logger.info("=== 平衡后 Label 分布 ===")
    total_samples_balanced = sum(len(samples) for samples in label_to_samples.values())
    for service_id in sorted(label_to_samples.keys()):
        count = len(label_to_samples[service_id])
        percentage = count / total_samples_balanced * 100
        if service_id == -1:
            logger.info(
                f"Label {service_id} (正常样本): {count} 个样本 ({percentage:.1f}%)"
            )
        else:
            service_name = next(
                (
                    name
                    for name, id_ in datapack.metadata.service_name_to_id.items()
                    if id_ == service_id
                ),
                f"Unknown_{service_id}",
            )
            logger.info(
                f"Label {service_id} ({service_name}): {count} 个样本 ({percentage:.1f}%)"
            )

    # Split train/test
    train_ratio = config.get("training.train_ratio")
    train_samples, test_samples = [], []

    train_label_counts = defaultdict(int)
    test_label_counts = defaultdict(int)

    for label, samples in label_to_samples.items():
        label_samples = samples.copy()
        random.shuffle(label_samples)
        split_idx = int(len(label_samples) * train_ratio)
        train_label_samples = label_samples[:split_idx]
        test_label_samples = label_samples[split_idx:]

        train_samples.extend(train_label_samples)
        test_samples.extend(test_label_samples)

        train_label_counts[label] = len(train_label_samples)
        test_label_counts[label] = len(test_label_samples)

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # 打印训练集和测试集的 label 分布
    logger.info("=== 训练集 Label 分布 ===")
    for service_id in sorted(train_label_counts.keys()):
        count = train_label_counts[service_id]
        percentage = count / len(train_samples) * 100
        if service_id == -1:
            logger.info(
                f"Label {service_id} (正常样本): {count} 个样本 ({percentage:.1f}%)"
            )
        else:
            service_name = next(
                (
                    name
                    for name, id_ in datapack.metadata.service_name_to_id.items()
                    if id_ == service_id
                ),
                f"Unknown_{service_id}",
            )
            logger.info(
                f"Label {service_id} ({service_name}): {count} 个样本 ({percentage:.1f}%)"
            )

    logger.info("=== 测试集 Label 分布 ===")
    for service_id in sorted(test_label_counts.keys()):
        count = test_label_counts[service_id]
        percentage = count / len(test_samples) * 100 if len(test_samples) > 0 else 0
        if service_id == -1:
            logger.info(
                f"Label {service_id} (正常样本): {count} 个样本 ({percentage:.1f}%)"
            )
        else:
            service_name = next(
                (
                    name
                    for name, id_ in datapack.metadata.service_name_to_id.items()
                    if id_ == service_id
                ),
                f"Unknown_{service_id}",
            )
            logger.info(
                f"Label {service_id} ({service_name}): {count} 个样本 ({percentage:.1f}%)"
            )

    logger.info(
        f"Loaded {len(train_samples)} training and {len(test_samples)} test time series"
    )
    return train_samples, test_samples, datapack.metadata


def create_timeseries_data_loaders(
    train_samples: List[TimeSeriesDataSample],
    test_samples: List[TimeSeriesDataSample],
    metadata: DatasetMetadata,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    batch_size = config.get("training.batch_size")
    window_size = config.get("datasets")[config.get("dataset")]["sample_interval"]
    step_size = config.get("datasets")[config.get("dataset")]["sample_step"]

    train_dataset = TimeWindowDataset(
        train_samples, metadata, window_size, step_size, shuffle=True
    )
    test_dataset = TimeWindowDataset(
        test_samples, metadata, window_size, step_size, shuffle=False
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


def setup_model_config(config: Config, metadata: DatasetMetadata):
    """Setup model configuration with metadata"""
    config.set("node_num", len(metadata.services))
    config.set("event_num", len(metadata.log_templates) + 1)
    config.set("metric_num", len(metadata.metrics))


app = typer.Typer()


@app.command()
def train(
    config_file: str = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Custom experiment name"),
    checkpoint_freq: int = typer.Option(5, help="Save checkpoint every N epochs"),
) -> None:
    config = Config(config_file)

    try:
        seed_everything(config.get("training.random_seed"))
        device = get_device(config.get("training.gpu"))

        train_samples, test_samples, metadata = load_timeseries_data(config)
        train_loader, test_loader = create_timeseries_data_loaders(
            train_samples, test_samples, metadata, config
        )

        setup_model_config(config, metadata)

        model = BaseModel(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=str(device),
            config=config,
            experiment_name=experiment_name,
        )

        model.fit(
            train_loader,
            test_loader,
            evaluation_epoch=config.get("training.evaluation_epoch"),
            checkpoint_frequency=checkpoint_freq,
        )

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


@app.command()
def resume(
    experiment_name: str = typer.Argument(..., help="Name of the experiment to resume"),
    config_file: str = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    checkpoint_path: Optional[str] = typer.Option(
        None, "--checkpoint", help="Specific checkpoint path"
    ),
    additional_epochs: Optional[int] = typer.Option(
        None, "--epochs", help="Additional epochs to train"
    ),
    lr: Optional[float] = typer.Option(None, "--lr", help="New learning rate"),
    gpu: Optional[bool] = typer.Option(None, help="Use GPU override"),
    load_best: bool = typer.Option(
        False, "--best/--latest", help="Load best vs latest checkpoint"
    ),
    checkpoint_freq: int = typer.Option(5, help="Save checkpoint every N epochs"),
) -> None:
    config = Config(config_file)

    if lr is not None:
        config.set("training.lr", lr)
    if gpu is not None:
        config.set("training.gpu", gpu)

    try:
        seed_everything(config.get("training.random_seed"))
        device = get_device(config.get("training.gpu"))

        train_samples, test_samples, metadata = load_timeseries_data(config)
        train_loader, test_loader = create_timeseries_data_loaders(
            train_samples, test_samples, metadata, config
        )

        setup_model_config(config, metadata)

        # Extend epochs if specified
        if additional_epochs is not None:
            current_epochs = config.get("training.epochs")
            config.set("training.epochs", current_epochs + additional_epochs)
            logger.info(
                f"Extended training from {current_epochs} to {current_epochs + additional_epochs} epochs"
            )

        logger.info(f"Loading model from experiment: {experiment_name}")
        model = BaseModel.from_experiment(
            experiment_name=experiment_name,
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=str(device),
            config=config,
            checkpoint_path=checkpoint_path,
            load_best=load_best,
        )

        model.fit(
            train_loader,
            test_loader,
            evaluation_epoch=config.get("training.evaluation_epoch"),
            checkpoint_frequency=checkpoint_freq,
            resume_from_checkpoint=checkpoint_path,
        )

        logger.info("Training resumed and completed successfully!")

    except Exception as e:
        logger.error(f"Failed to resume training: {str(e)}")
        raise


@app.command()
def inference(
    experiment_name: str = typer.Argument(..., help="Name of the experiment"),
    config_file: str = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    checkpoint_path: Optional[str] = typer.Option(
        None, "--checkpoint", help="Specific checkpoint path"
    ),
    load_best: bool = typer.Option(
        True, "--best/--latest", help="Load best vs latest checkpoint"
    ),
    dataset: Optional[str] = typer.Option(None, help="Dataset name override"),
    gpu: Optional[bool] = typer.Option(None, help="Use GPU override"),
    save_results: bool = typer.Option(
        True, "--save/--no-save", help="Save inference results"
    ),
    use_test_data: bool = typer.Option(
        True, "--test/--train", help="Use test vs train data"
    ),
) -> None:
    config = Config(config_file)
    if dataset:
        config.set("dataset", dataset)
    if gpu is not None:
        config.set("training.gpu", gpu)

    try:
        seed_everything(config.get("training.random_seed"))
        device = get_device(config.get("training.gpu"))

        train_samples, test_samples, metadata = load_timeseries_data(config)
        train_loader, test_loader = create_timeseries_data_loaders(
            train_samples, test_samples, metadata, config
        )

        setup_model_config(config, metadata)

        logger.info(f"Loading model from experiment: {experiment_name}")
        model = BaseModel.from_experiment(
            experiment_name=experiment_name,
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=str(device),
            config=config,
            checkpoint_path=checkpoint_path,
            load_best=load_best,
        )

        data_loader = test_loader if use_test_data else train_loader
        data_type = "test" if use_test_data else "train"

        logger.info(f"Running inference on {data_type} data...")
        results = model.inference(data_loader, save_results=save_results)

        logger.info("Inference Results:")
        logger.info(f"Total samples: {results['total_samples']}")
        logger.info("Metrics:")
        for metric_name, value in results["metrics"].items():
            logger.info(f"  {metric_name}: {value:.4f}")

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


@app.command()
def list_experiments(
    result_dir: str = typer.Option("result", help="Result directory"),
) -> None:
    """List all available experiments"""

    result_path = Path(result_dir)
    if not result_path.exists():
        logger.error(f"Result directory not found: {result_dir}")
        return

    experiments = [
        exp_dir.name
        for exp_dir in result_path.iterdir()
        if exp_dir.is_dir() and (exp_dir / "config.json").exists()
    ]

    if experiments:
        logger.info(f"Found {len(experiments)} experiments:")
        for exp_name in sorted(experiments):
            logger.info(f"  - {exp_name}")
    else:
        logger.info("No experiments found")


@app.command()
def list_checkpoints(
    experiment_name: str = typer.Argument(..., help="Name of the experiment"),
    result_dir: str = typer.Option("result", help="Result directory"),
) -> None:
    """List all checkpoints for an experiment"""

    try:
        from src.eadro.experiment_manager import ExperimentManager

        exp_manager = ExperimentManager.load_experiment(experiment_name, result_dir)
        checkpoints = exp_manager.list_checkpoints()

        if checkpoints:
            logger.info(
                f"Found {len(checkpoints)} checkpoints for experiment '{experiment_name}':"
            )
            for cp in sorted(checkpoints, key=lambda x: x["epoch"]):
                epoch = cp["epoch"]
                timestamp = cp["timestamp"]
                is_best = " (BEST)" if cp.get("is_best", False) else ""
                metrics = cp.get("metrics", {})
                hr1 = metrics.get("HR@1", "N/A")
                logger.info(
                    f"  Epoch {epoch:4d}: HR@1={hr1:.4f} [{timestamp}]{is_best}"
                )
        else:
            logger.info(f"No checkpoints found for experiment '{experiment_name}'")

    except Exception as e:
        logger.error(f"Failed to list checkpoints: {str(e)}")


if __name__ == "__main__":
    app()
