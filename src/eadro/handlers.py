"""
Handlers for integrating EADRO model with the universal experiment manager
"""

import torch
from torch.optim.adam import Adam
from src.exp.controller import (
    ModelHandler,
    DataHandler,
    TrainingHandler,
    InferenceHandler,
    MetricsDict,
    ConfigDict,
    PredictionResult,
)
from src.exp.config import Config
from .base import BaseModel
from pathlib import Path
from loguru import logger
from typing import List, Optional, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
from src.preprocessing.base import DataSample, DatasetMetadata
from typing import Counter
import pickle
import dgl


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

    # Log balanced distribution
    logger.info("Balanced label distribution:")
    for label, samples in label_to_samples.items():
        logger.info(f"Service {label}: {len(samples)} samples")

    # Split each label's samples into train and test first
    train_ratio = config.get("training.train_ratio")
    train_label_to_samples = defaultdict(list)
    test_samples = []

    for label, samples in label_to_samples.items():
        label_samples = samples.copy()
        random.shuffle(label_samples)

        # Split this label's samples
        split_idx = int(len(label_samples) * train_ratio)
        train_label_to_samples[label].extend(label_samples[:split_idx])
        test_samples.extend(label_samples[split_idx:])

    # Balance training set by finding minimum count and downsampling all labels
    train_counts = [len(samples) for samples in train_label_to_samples.values()]
    if train_counts:
        min_train_count = min(train_counts)
        logger.info(f"Balancing training set to {min_train_count} samples per label")

        train_samples = []
        for label, samples in train_label_to_samples.items():
            # Downsample to minimum count
            balanced_samples = samples[:min_train_count]
            train_samples.extend(balanced_samples)
            logger.info(
                f"Training set - Service {label}: {len(balanced_samples)} samples (original: {len(samples)})"
            )
    else:
        train_samples = []

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


def collate_fn(
    batch: List[Tuple[dgl.DGLGraph, int]],
) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class ChunkDataset(Dataset):
    def __init__(
        self,
        samples: List[DataSample],
        metadata: DatasetMetadata,
    ):
        self.metadata = metadata
        self.node_num = len(metadata.services)

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


def create_data_loaders(
    train_samples: List[DataSample],
    test_samples: List[DataSample],
    metadata: DatasetMetadata,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    # Create datasets with shuffling for training data
    train_dataset = ChunkDataset(train_samples, metadata)
    test_dataset = ChunkDataset(test_samples, metadata)

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


class OptimizerAdapter:
    """Adapter to make torch optimizers compatible with HasStateDict protocol"""

    def __init__(self, optimizer: torch.optim.Optimizer):  # type: ignore
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get optimizer state dictionary"""
        return self.optimizer.state_dict()

    def load_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ) -> object:
        """Load state dictionary into optimizer"""
        self.optimizer.load_state_dict(state_dict)
        return self

    def zero_grad(self) -> None:
        """Zero gradients"""
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Optimization step"""
        self.optimizer.step()

    @property
    def param_groups(self):
        """Access to parameter groups"""
        return self.optimizer.param_groups


class EadroModelHandler(ModelHandler[BaseModel]):
    """Model handler for EADRO models"""

    def __init__(self, device: str):
        self.device = device

    def save_model(self, model: BaseModel, path: str) -> None:
        """Save EADRO model to disk"""
        torch.save(model.model.state_dict(), path)

    def load_model(self, path: str) -> BaseModel:
        """Load EADRO model from disk - requires separate model creation"""
        # This is a placeholder - actual loading requires model architecture
        raise NotImplementedError(
            "Model loading requires separate architecture creation"
        )

    def get_model_state(self, model: BaseModel) -> Dict[str, torch.Tensor]:
        """Get model state dictionary"""
        return model.model.state_dict()

    def load_model_state(
        self, model: BaseModel, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Load state dictionary into model"""
        model.model.load_state_dict(state_dict)


class EadroDataHandler(DataHandler[Tuple[DataLoader, DataLoader, DatasetMetadata]]):
    """Data handler for EADRO dataset loading and preprocessing"""

    def __init__(self):
        self.metadata: Optional[DatasetMetadata] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def prepare_data(
        self, config: Config
    ) -> Tuple[DataLoader, DataLoader, DatasetMetadata]:
        # Load preprocessed samples and metadata
        train_samples, test_samples, metadata = load_data(config)

        # Create data loaders
        train_loader, test_loader = create_data_loaders(
            train_samples, test_samples, metadata, config
        )

        # Store for later use
        self.metadata = metadata
        self.train_loader = train_loader
        self.test_loader = test_loader

        return train_loader, test_loader, metadata

    def get_data_info(self) -> ConfigDict:
        """Get information about the dataset"""
        if self.metadata is None:
            return {}

        return {
            "num_services": len(self.metadata.services),
            "num_log_templates": len(self.metadata.log_templates),
            "num_metrics": len(self.metadata.metrics),
            "num_edges": len(self.metadata.service_calling_edges),
            "train_batches": len(self.train_loader) if self.train_loader else 0,
            "test_batches": len(self.test_loader) if self.test_loader else 0,
        }


class EadroTrainingHandler(
    TrainingHandler[
        BaseModel, Tuple[DataLoader, DataLoader, DatasetMetadata], OptimizerAdapter
    ]
):
    """Training handler for EADRO model"""

    def __init__(self, device: str):
        self.device = device

    def train_epoch(
        self,
        model: BaseModel,
        data: Tuple[DataLoader, DataLoader, DatasetMetadata],
        optimizer: OptimizerAdapter,
        **kwargs,
    ) -> MetricsDict:
        """Train model for one epoch"""
        train_loader, _, _ = data
        model.model.train()

        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (graph, ground_truth) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            result = model.model.forward(graph.to(self.device), ground_truth)
            loss = result["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0

        return {"train_loss": avg_loss, "train_batches": batch_count}

    def validate(
        self,
        model: BaseModel,
        data: Tuple[DataLoader, DataLoader, DatasetMetadata],
        **kwargs,
    ) -> MetricsDict:
        _, test_loader, _ = data
        return model.evaluate(test_loader, datatype="Validation")

    def setup_optimizer(self, model: BaseModel, config: Config) -> OptimizerAdapter:
        """Setup optimizer for training"""
        lr = config.get("training.lr")
        adam_optimizer = Adam(model.model.parameters(), lr=lr)
        return OptimizerAdapter(adam_optimizer)


class EadroInferenceHandler(
    InferenceHandler[BaseModel, Tuple[DataLoader, DataLoader, DatasetMetadata]]
):
    def __init__(self, device: str):
        self.device = device

    def predict(
        self,
        model: BaseModel,
        data: Tuple[DataLoader, DataLoader, DatasetMetadata],
    ) -> torch.Tensor:
        _, test_loader, _ = data
        model.model.eval()

        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for graph, ground_truths in test_loader:
                result = model.model.forward(graph.to(self.device), ground_truths)
                predictions = result["y_pred"]
                probabilities = result["pred_prob"]

                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)

        return torch.tensor(all_predictions)

    def postprocess_results(
        self, predictions: torch.Tensor, **kwargs
    ) -> PredictionResult:
        predictions_list = predictions.tolist()

        normal_count = sum(1 for pred in predictions_list if pred[0] == -1)
        anomaly_count = len(predictions_list) - normal_count

        return {
            "predictions": predictions_list,
            "total_samples": len(predictions_list),
            "normal_samples": normal_count,
            "anomaly_samples": anomaly_count,
            "anomaly_rate": anomaly_count / len(predictions_list)
            if predictions_list
            else 0.0,
        }


def create_eadro_model(
    event_num: int, metric_num: int, node_num: int, device: str, config: Config
) -> BaseModel:
    return BaseModel(
        event_num=event_num,
        metric_num=metric_num,
        node_num=node_num,
        device=device,
        config=config,
    )
