"""
EADRO 项目的实验处理器实现
基于 UniversalExperimentManager SDK 的具体实现
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import ndcg_score
from loguru import logger
import numpy as np
import dgl
import random

from .experiment_sdk import (
    DataHandler,
    TrainingHandler,
    InferenceHandler,
    PyTorchModelHandler,
)
from .model import MainModel
from .config import Config
from ..preprocessing.base import (
    DatasetMetadata,
    TimeSeriesDataSample,
    TimeSeriesDatapack,
    DataSample,
)


def get_device(use_gpu: bool = True):
    """获取可用的设备"""
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


class EadroModelHandler(PyTorchModelHandler):
    """EADRO 模型处理器，继承自 PyTorchModelHandler"""

    def __init__(self):
        super().__init__()

    def create_model(
        self,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: str,
        config: Config,
    ) -> torch.nn.Module:
        model = MainModel(event_num, metric_num, node_num, device, config)
        model.to(device)
        return model


class EadroDataHandler(DataHandler):
    """EADRO 数据处理器"""

    def __init__(
        self,
        train_samples: List[TimeSeriesDataSample],
        test_samples: List[TimeSeriesDataSample],
        metadata: DatasetMetadata,
    ):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.metadata = metadata
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def prepare_data(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = config.get("training.batch_size")
        dataset_name = config.get("dataset")
        datasets_config = config.get("datasets")
        dataset_config = datasets_config.get(dataset_name, {})

        window_size = dataset_config.get("sample_interval", 10)
        step_size = dataset_config.get("sample_step", 1)

        train_dataset = TimeWindowDataset(
            self.train_samples, self.metadata, window_size, step_size, shuffle=True
        )
        test_dataset = TimeWindowDataset(
            self.test_samples, self.metadata, window_size, step_size, shuffle=False
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

        self.train_loader = train_loader
        self.test_loader = test_loader

        return train_loader, test_loader

    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        return {
            "train_samples": len(self.train_samples),
            "test_samples": len(self.test_samples),
            "services": len(self.metadata.services),
            "service_calling_edges": len(self.metadata.service_calling_edges),
            "metadata": {
                "services": self.metadata.services,
                "service_name_to_id": self.metadata.service_name_to_id,
            },
        }


class EadroTrainingHandler(TrainingHandler):
    """EADRO 训练处理器"""

    def __init__(self, device: str, config: Config):
        self.device = device
        self.config = config

        # 从配置中获取学习率调度器参数
        lr_scheduler_config = config.get("training.lr_scheduler")
        self.lr_scheduler_type = lr_scheduler_config.get("type", "none").lower()
        self.lr_step_size = lr_scheduler_config.get("step_size")
        self.lr_gamma = lr_scheduler_config.get("gamma")
        self.lr_warmup_epochs = lr_scheduler_config.get("warmup_epochs")
        self.lr_min = lr_scheduler_config.get("min_lr")
        self.patience = config.get("training.patience")

        self.best_hr1 = -1
        self.worse_count = 0
        self.pre_loss = float("inf")

    def setup_optimizer(self, model: torch.nn.Module, config: Config) -> Any:
        lr = config.get("training.lr")
        return torch.optim.Adam(model.parameters(), lr=lr)  # type: ignore

    def _create_lr_scheduler(self, optimizer: Any, epochs: int) -> Optional[Any]:
        """创建学习率调度器"""
        if self.lr_scheduler_type == "none":
            return None
        elif self.lr_scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
            )
        elif self.lr_scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.lr_gamma
            )
        elif self.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif self.lr_scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.lr_gamma, patience=5
            )
        else:
            logger.warning(
                f"Unknown scheduler type: {self.lr_scheduler_type}, using none"
            )
            return None

    def _warmup_lr(self, optimizer: Any, epoch: int) -> None:
        """学习率预热"""
        if epoch <= self.lr_warmup_epochs and self.lr_warmup_epochs > 0:
            warmup_factor = epoch / self.lr_warmup_epochs
            lr = self.config.get("training.lr") * warmup_factor
            lr = max(lr, self.lr_min)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def train_epoch(
        self, model: torch.nn.Module, data: DataLoader, optimizer: Any, **kwargs
    ) -> Dict[str, float]:
        """训练一个epoch"""
        epoch = kwargs.get("epoch", 0)
        scheduler = kwargs.get("scheduler")

        model.train()
        batch_cnt, epoch_loss = 0, 0.0
        epoch_time_start = time.time()

        # 应用学习率预热
        self._warmup_lr(optimizer, epoch)

        for graph, groundtruth in data:
            optimizer.zero_grad()
            result = model.forward(graph.to(self.device), groundtruth)
            loss = result["loss"]
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
            batch_cnt += 1

        # 在预热期后调用调度器（除了 ReduceLROnPlateau）
        if (
            scheduler is not None
            and epoch > self.lr_warmup_epochs
            and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        ):
            scheduler.step()

        epoch_time_elapsed = time.time() - epoch_time_start
        avg_epoch_loss = epoch_loss / batch_cnt
        current_lr = optimizer.param_groups[0]["lr"]

        # 早停机制
        if avg_epoch_loss > self.pre_loss:
            self.worse_count += 1
        else:
            self.worse_count = 0
        self.pre_loss = avg_epoch_loss

        return {
            "loss": avg_epoch_loss,
            "lr": current_lr,
            "time": epoch_time_elapsed,
            "worse_count": self.worse_count,
        }

    def validate(
        self, model: torch.nn.Module, data: DataLoader, **kwargs
    ) -> Dict[str, float]:
        model.eval()

        all_prob = []
        all_result = []
        all_labels = []

        with torch.no_grad():
            for graph, groundtruth in data:
                result = model.forward(graph.to(self.device), groundtruth)
                assert "pred_prob" in result and "y_pred" in result, (
                    "Model forward must return 'pred_prob' and 'y_pred'"
                )
                prob = result["pred_prob"]
                pred_result = result["y_pred"]

                all_prob.append(prob)
                all_result.append(pred_result)
                all_labels.append(groundtruth.cpu().detach().numpy())

        all_prob = np.concatenate(all_prob, axis=0)
        all_result = np.concatenate(all_result, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 计算指标
        metrics = self._calculate_metrics(all_prob, all_result, all_labels)

        return metrics

    def _calculate_metrics(
        self, all_prob: np.ndarray, all_result: np.ndarray, all_labels: np.ndarray
    ) -> Dict[str, float]:
        """计算评估指标"""
        total_sample = len(all_labels)

        # HR@k 计算
        hr1_total, hr3_total, hr5_total = 0, 0, 0

        # NDCG@k 计算
        ndcg1_list, ndcg3_list, ndcg5_list = [], [], []

        for i in range(total_sample):
            prob = all_prob[i]
            label = all_labels[i]

            # 获取排序索引
            sorted_idx = np.argsort(prob)[::-1]

            # HR@k
            if label in sorted_idx[:1]:
                hr1_total += 1
            if label in sorted_idx[:3]:
                hr3_total += 1
            if label in sorted_idx[:5]:
                hr5_total += 1

            # NDCG@k
            true_relevance = np.zeros(len(prob))
            true_relevance[label] = 1

            try:
                ndcg1 = ndcg_score(
                    true_relevance.reshape(1, -1), prob.reshape(1, -1), k=1
                )
                ndcg3 = ndcg_score(
                    true_relevance.reshape(1, -1), prob.reshape(1, -1), k=3
                )
                ndcg5 = ndcg_score(
                    true_relevance.reshape(1, -1), prob.reshape(1, -1), k=5
                )

                ndcg1_list.append(ndcg1)
                ndcg3_list.append(ndcg3)
                ndcg5_list.append(ndcg5)
            except Exception:
                # 处理 NDCG 计算异常
                ndcg1_list.append(0.0)
                ndcg3_list.append(0.0)
                ndcg5_list.append(0.0)

        return {
            "HR@1": float(hr1_total / total_sample),
            "HR@3": float(hr3_total / total_sample),
            "HR@5": float(hr5_total / total_sample),
            "NDCG@1": float(np.mean(ndcg1_list)),
            "NDCG@3": float(np.mean(ndcg3_list)),
            "NDCG@5": float(np.mean(ndcg5_list)),
        }


class EadroInferenceHandler(InferenceHandler):
    def __init__(self, device: str):
        self.device = device

    def predict(
        self, model: torch.nn.Module, data: DataLoader, **kwargs
    ) -> Dict[str, Any]:
        model.eval()

        all_predictions = []
        all_ground_truths = []
        all_probabilities = []

        with torch.no_grad():
            for graph, groundtruth in data:
                result = model.forward(graph.to(self.device), groundtruth)

                prob = result["prob"].cpu().detach().numpy()
                pred_result = result["result"].cpu().detach().numpy()

                all_probabilities.append(prob)
                all_predictions.append(pred_result)
                all_ground_truths.append(groundtruth.cpu().detach().numpy())

        return {
            "probabilities": np.concatenate(all_probabilities, axis=0),
            "predictions": np.concatenate(all_predictions, axis=0),
            "ground_truths": np.concatenate(all_ground_truths, axis=0),
        }

    def postprocess_results(
        self, predictions: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """后处理推理结果"""
        probabilities = predictions["probabilities"]
        pred_results = predictions["predictions"]
        ground_truths = predictions["ground_truths"]

        total_samples = len(ground_truths)

        # 计算指标
        hr1_total, hr3_total, hr5_total = 0, 0, 0
        ndcg1_list, ndcg3_list, ndcg5_list = [], [], []

        for i in range(total_samples):
            prob = probabilities[i]
            label = ground_truths[i]

            # 获取排序索引
            sorted_idx = np.argsort(prob)[::-1]

            # HR@k
            if label in sorted_idx[:1]:
                hr1_total += 1
            if label in sorted_idx[:3]:
                hr3_total += 1
            if label in sorted_idx[:5]:
                hr5_total += 1

            # NDCG@k
            true_relevance = np.zeros(len(prob))
            true_relevance[label] = 1

            try:
                ndcg1 = ndcg_score(
                    true_relevance.reshape(1, -1), prob.reshape(1, -1), k=1
                )
                ndcg3 = ndcg_score(
                    true_relevance.reshape(1, -1), prob.reshape(1, -1), k=3
                )
                ndcg5 = ndcg_score(
                    true_relevance.reshape(1, -1), prob.reshape(1, -1), k=5
                )

                ndcg1_list.append(ndcg1)
                ndcg3_list.append(ndcg3)
                ndcg5_list.append(ndcg5)
            except Exception:
                ndcg1_list.append(0.0)
                ndcg3_list.append(0.0)
                ndcg5_list.append(0.0)

        metrics = {
            "HR@1": float(hr1_total / total_samples),
            "HR@3": float(hr3_total / total_samples),
            "HR@5": float(hr5_total / total_samples),
            "NDCG@1": float(np.mean(ndcg1_list)),
            "NDCG@3": float(np.mean(ndcg3_list)),
            "NDCG@5": float(np.mean(ndcg5_list)),
        }

        return {
            "total_samples": total_samples,
            "metrics": metrics,
            "predictions": pred_results,
            "probabilities": probabilities,
            "ground_truths": ground_truths,
        }
