"""
Handlers for integrating EADRO model with the universal experiment manager
"""

import glob
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Counter, Dict, List, Optional, Tuple

import dgl
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import ndcg_score
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset

from src.exp.config import Config
from src.exp.controller import (
    ConfigDict,
    DataHandler,
    InferenceHandler,
    MetricsDict,
    ModelHandler,
    PredictionResult,
    TrainingHandler,
)
from src.preprocessing.base import DataSample, DatasetMetadata

from .model import MainModel


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
            logs_tensor = torch.FloatTensor(sample.log)

            metrics_array = np.array(sample.metric)
            if np.isnan(metrics_array).any():
                metrics_array = np.nan_to_num(metrics_array, nan=0.0)
            metrics_tensor = torch.FloatTensor(metrics_array)

            traces_array = np.array(sample.trace)
            if np.isnan(traces_array).any():
                logger.warning(
                    f"Found NaN values in traces array for sample {len(self.graphs)}"
                )
                traces_array = np.nan_to_num(traces_array, nan=0.0)

            traces_tensor = torch.FloatTensor(traces_array)

            logs_array = np.array(sample.log)
            if np.isnan(logs_array).any():
                logs_array = np.nan_to_num(logs_array, nan=0.0)
                logs_tensor = torch.FloatTensor(logs_array)

            assert not torch.isnan(logs_tensor).any(), (
                f"logs data contains NaN values in sample {len(self.graphs)}"
            )
            assert not torch.isnan(metrics_tensor).any(), (
                f"metrics data contains NaN values in sample {len(self.graphs)}"
            )
            assert not torch.isnan(traces_tensor).any(), (
                f"traces data contains NaN values in sample {len(self.graphs)}"
            )

            # Additional check for infinite values
            assert torch.isfinite(logs_tensor).all(), (
                f"logs data contains infinite values in sample {len(self.graphs)}"
            )
            assert torch.isfinite(metrics_tensor).all(), (
                f"metrics data contains infinite values in sample {len(self.graphs)}"
            )
            assert torch.isfinite(traces_tensor).all(), (
                f"traces data contains infinite values in sample {len(self.graphs)}"
            )

            graph.ndata["logs"] = logs_tensor
            graph.ndata["metrics"] = metrics_tensor
            graph.ndata["traces"] = traces_tensor

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


class EadroDataHandler(DataHandler[Tuple[DataLoader, DataLoader, DatasetMetadata]]):
    def __init__(self):
        self.metadata: Optional[DatasetMetadata] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def create_data_loaders(
        self,
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

    def load_data(
        self,
        config: Config,
        dataset_folder: str,
        train_split: str = "train",
        test_split: str = "test",
        test_dataset_folder: Optional[str] = None,
    ) -> Tuple[List[DataSample], List[DataSample], DatasetMetadata]:
        dataset_name = config.get("dataset")

        # 如果没有指定test_dataset_folder，则使用相同的dataset_folder
        actual_test_folder = (
            test_dataset_folder if test_dataset_folder is not None else dataset_folder
        )

        # 使用统一的 metadata 目录
        metadata_dir = Path(config.get("paths.metadata"))

        # 新格式：{dataset}_{dataset_folder}_{label}_samples_batch_*.pkl
        train_batch_pattern = str(
            metadata_dir
            / f"{dataset_name}_{dataset_folder}_{train_split}_samples_batch_*.pkl"
        )
        test_batch_pattern = str(
            metadata_dir
            / f"{dataset_name}_{actual_test_folder}_{test_split}_samples_batch_*.pkl"
        )

        # Load training data
        train_batch_files = glob.glob(train_batch_pattern)

        # Load test data
        test_batch_files = glob.glob(test_batch_pattern)

        # Load training samples
        train_samples = []
        if train_batch_files:
            logger.info(f"Found {len(train_batch_files)} training batch files to load")
            train_batch_files.sort()  # Ensure consistent ordering

            for batch_file in train_batch_files:
                logger.info(f"Loading training batch file: {batch_file}")
                with open(batch_file, "rb") as f:
                    batch_samples = pickle.load(f)
                    train_samples.extend(batch_samples)
                    logger.info(
                        f"Loaded {len(batch_samples)} training samples from {batch_file}"
                    )
            logger.info(f"Total training samples loaded: {len(train_samples)}")

        # Load test samples
        test_samples = []
        if test_batch_files:
            logger.info(f"Found {len(test_batch_files)} test batch files to load")
            test_batch_files.sort()  # Ensure consistent ordering

            for batch_file in test_batch_files:
                logger.info(f"Loading test batch file: {batch_file}")
                with open(batch_file, "rb") as f:
                    batch_samples = pickle.load(f)
                    test_samples.extend(batch_samples)
                    logger.info(
                        f"Loaded {len(batch_samples)} test samples from {batch_file}"
                    )
            logger.info(f"Total test samples loaded: {len(test_samples)}")

        # Check if we have any data
        if not train_samples and not test_samples:
            raise FileNotFoundError(
                f"No sample files found. Expected training files matching '{train_batch_pattern}' "
                f"or test files matching '{test_batch_pattern}'"
            )

        # Load metadata - 总是加载train的metadata（无论当前处理的是train还是test）
        metadata_path = (
            metadata_dir / f"{dataset_name}_{dataset_folder}_train_metadata.pkl"
        )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                f"Please ensure you have processed the train dataset for folder '{dataset_folder}' first."
            )

        metadata = DatasetMetadata.from_pkl(str(metadata_path))
        if metadata is None:
            raise ValueError(f"Failed to load metadata from {metadata_path}")

        # Apply balancing to training set if requested
        if config.get("training.balance_train_set") and train_samples:
            # Group training samples by label
            train_label_to_samples = defaultdict(list)
            for sample in train_samples:
                label = sample.get_gt_service_id(metadata.service_name_to_id)
                train_label_to_samples[label].append(sample)

            # Balance by finding minimum count and downsampling
            train_counts = [len(samples) for samples in train_label_to_samples.values()]
            if train_counts:
                min_train_count = min(train_counts)
                logger.info(
                    f"Balancing training set to {min_train_count} samples per label"
                )

                balanced_train_samples = []
                for label, samples in train_label_to_samples.items():
                    balanced_samples = samples[: min_train_count * 20]
                    balanced_train_samples.extend(balanced_samples)
                    logger.info(
                        f"Training set - Service {label}: {len(balanced_samples)} samples (original: {len(samples)})"
                    )
                train_samples = balanced_train_samples

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

    def load_data_by_dataset_id(
        self,
        config: Config,
        dataset_id: int,
        train_split: str = "train",
        test_split: str = "test",
    ) -> Tuple[List[DataSample], List[DataSample], DatasetMetadata]:
        """
        Load data using dataset ID from RCABench platform

        Args:
            config: Configuration object
            dataset_id: Dataset ID from RCABench platform
            train_split: Training data split identifier
            test_split: Test data split identifier

        Returns:
            Tuple of (train_samples, test_samples, metadata)
        """
        from rcabench.openapi import DatasetsApi
        from rcabench_platform.v2.clients.rcabench_ import RCABenchClient
        from rcabench_platform.v2.datasets.rcabench import valid
        from rcabench_platform.v2.sources.convert import convert_datapack
        from rcabench_platform.v2.sources.rcabench import RcabenchDatapackLoader

        from src.preprocessing.processor import process_datapack_parallel

        # Get dataset information from RCABench platform
        with RCABenchClient() as client:
            api = DatasetsApi(client)
            resp = api.api_v2_datasets_id_get(id=dataset_id, include_injections=True)
            assert resp.data is not None
            assert resp.data.injections is not None

        datapacks_injections = resp.data.injections
        logger.info(
            f"Found {len(datapacks_injections)} datapacks for dataset ID {dataset_id}"
        )

        dataset_name = config.get("dataset")
        dataset_config = config.get(f"datasets.{dataset_name}")

        sample_interval = int(dataset_config.get("sample_interval", 30))
        sample_step = int(dataset_config.get("sample_step", 30))
        log_files = dataset_config.get("log_files", [])
        metric_files = dataset_config.get("metric_files", [])
        trace_files = dataset_config.get("trace_files", [])

        # Process each datapack and collect samples
        all_train_samples = []
        all_test_samples = []
        metadata = None

        for datapack_info in datapacks_injections:
            injection_name = datapack_info.injection_name

            # Set up paths following the structure from your code
            input_path = Path("data") / "rcabench_dataset" / injection_name
            converted_input_path = input_path / "converted"

            # Validate and convert if needed
            _, _valid = valid(input_path)
            if not _valid:
                logger.warning(f"Skipping invalid datapack: {injection_name}")
                continue

            # Convert datapack if not already converted
            convert_datapack(
                loader=RcabenchDatapackLoader(
                    src_folder=input_path, datapack=injection_name
                ),
                dst_folder=converted_input_path,
                skip_finished=True,
            )

            # Create metadata from first valid datapack
            if metadata is None:
                from src.preprocessing.processor import Processor

                processor = Processor({}, conf="settings.toml")
                processor.datapacks = [converted_input_path]
                processor.derive_files()
                metadata = processor.create_metadata()
                logger.info(f"Created metadata from datapack: {injection_name}")

            # Process the converted datapack
            samples = process_datapack_parallel(
                converted_input_path,
                dataset_name,
                sample_interval,
                sample_step,
                metadata,
                log_files,
                metric_files,
                trace_files,
                "drain.ini",
                "cache/drain/temp",
            )

            # Split samples based on abnormal flag or use split parameter
            if train_split == "train" and test_split == "test":
                # Split based on abnormal flag
                train_samples = [s for s in samples if not s.abnormal]
                test_samples = [s for s in samples if s.abnormal]
            else:
                # Use all samples for the specified split
                if train_split == test_split:
                    train_samples = samples
                    test_samples = samples
                else:
                    train_samples = samples if train_split != "test" else []
                    test_samples = samples if test_split != "train" else []

            all_train_samples.extend(train_samples)
            all_test_samples.extend(test_samples)

            logger.info(
                f"Processed {injection_name}: {len(train_samples)} train, {len(test_samples)} test samples"
            )

        if metadata is None:
            raise ValueError(f"No valid datapacks found for dataset ID {dataset_id}")

        logger.info(
            f"Total samples for dataset ID {dataset_id}: {len(all_train_samples)} train, {len(all_test_samples)} test"
        )

        return all_train_samples, all_test_samples, metadata

    def load_train_data_by_dataset_id(
        self,
        config: Config,
        dataset_id: int,
    ) -> Tuple[List[DataSample], DatasetMetadata]:
        """
        Load training data using dataset ID from RCABench platform

        Args:
            config: Configuration object
            dataset_id: Dataset ID from RCABench platform for training

        Returns:
            Tuple of (train_samples, metadata)
        """
        from rcabench.openapi import DatasetsApi
        from rcabench_platform.v2.clients.rcabench_ import RCABenchClient
        from rcabench_platform.v2.datasets.rcabench import valid
        from rcabench_platform.v2.sources.convert import convert_datapack
        from rcabench_platform.v2.sources.rcabench import RcabenchDatapackLoader

        from src.preprocessing.processor import process_datapack_parallel

        # Get dataset information from RCABench platform
        with RCABenchClient() as client:
            api = DatasetsApi(client)
            resp = api.api_v2_datasets_id_get(id=dataset_id, include_injections=True)
            assert resp.data is not None
            assert resp.data.injections is not None

        datapacks_injections = resp.data.injections
        logger.info(
            f"Loading training data from dataset ID {dataset_id} with {len(datapacks_injections)} datapacks"
        )

        dataset_name = config.get("dataset")
        dataset_config = config.get(f"datasets.{dataset_name}")

        sample_interval = int(dataset_config.get("sample_interval", 30))
        sample_step = int(dataset_config.get("sample_step", 30))
        log_files = dataset_config.get("log_files", [])
        metric_files = dataset_config.get("metric_files", [])
        trace_files = dataset_config.get("trace_files", [])

        all_train_samples = []
        converted_datapacks = []

        # 先转换所有 datapacks，收集它们用于创建完整元数据
        for datapack_info in datapacks_injections:
            injection_name = datapack_info.injection_name

            # Set up paths following the structure from your code
            input_path = Path("data") / "rcabench_dataset" / injection_name
            converted_input_path = input_path / "converted"

            # Validate and convert if needed
            _, _valid = valid(input_path)
            if not _valid:
                logger.warning(f"Skipping invalid datapack: {injection_name}")
                continue

            # Convert datapack if not already converted
            convert_datapack(
                loader=RcabenchDatapackLoader(
                    src_folder=input_path, datapack=injection_name
                ),
                dst_folder=converted_input_path,
                skip_finished=True,
            )

            converted_datapacks.append(converted_input_path)

        if not converted_datapacks:
            raise ValueError(
                f"No valid datapacks found for training dataset ID {dataset_id}"
            )

        # 使用所有转换的 datapacks 创建完整元数据（等效于之前的方式）
        logger.info(
            f"Creating metadata from all {len(converted_datapacks)} datapacks..."
        )
        from src.preprocessing.processor import Processor

        processor = Processor({}, conf="settings.toml")
        processor.datapacks = converted_datapacks
        processor.derive_files()
        metadata = processor.create_metadata()
        logger.info("Created complete metadata from all training datapacks")

        # 现在处理每个 datapack 生成样本
        for converted_input_path in converted_datapacks:
            # Process the converted datapack - use ALL samples for training
            samples = process_datapack_parallel(
                converted_input_path,
                dataset_name,
                sample_interval,
                sample_step,
                metadata,
                log_files,
                metric_files,
                trace_files,
                "drain.ini",
                "cache/drain/temp",
            )

            all_train_samples.extend(samples)
            logger.info(
                f"Processed training datapack {converted_input_path.name}: {len(samples)} samples"
            )

        logger.info(
            f"Total training samples from dataset ID {dataset_id}: {len(all_train_samples)}"
        )
        return all_train_samples, metadata

    def load_test_data_by_dataset_id(
        self,
        config: Config,
        dataset_id: int,
        metadata: DatasetMetadata,
    ) -> List[DataSample]:
        """
        Load test data using dataset ID from RCABench platform

        Args:
            config: Configuration object
            dataset_id: Dataset ID from RCABench platform for testing
            metadata: Metadata from training data

        Returns:
            List of test samples
        """
        from rcabench.openapi import DatasetsApi
        from rcabench_platform.v2.clients.rcabench_ import RCABenchClient
        from rcabench_platform.v2.datasets.rcabench import valid
        from rcabench_platform.v2.sources.convert import convert_datapack
        from rcabench_platform.v2.sources.rcabench import RcabenchDatapackLoader

        from src.preprocessing.processor import process_datapack_parallel

        # Get dataset information from RCABench platform
        with RCABenchClient() as client:
            api = DatasetsApi(client)
            resp = api.api_v2_datasets_id_get(id=dataset_id, include_injections=True)
            assert resp.data is not None
            assert resp.data.injections is not None

        datapacks_injections = resp.data.injections
        logger.info(
            f"Loading test data from dataset ID {dataset_id} with {len(datapacks_injections)} datapacks"
        )

        dataset_name = config.get("dataset")
        dataset_config = config.get(f"datasets.{dataset_name}")

        sample_interval = int(dataset_config.get("sample_interval", 30))
        sample_step = int(dataset_config.get("sample_step", 30))
        log_files = dataset_config.get("log_files", [])
        metric_files = dataset_config.get("metric_files", [])
        trace_files = dataset_config.get("trace_files", [])

        all_test_samples = []

        for datapack_info in datapacks_injections:
            injection_name = datapack_info.injection_name

            # Set up paths following the structure from your code
            input_path = Path("data") / "rcabench_dataset" / injection_name
            converted_input_path = input_path / "converted"

            # Validate and convert if needed
            _, _valid = valid(input_path)
            if not _valid:
                logger.warning(f"Skipping invalid datapack: {injection_name}")
                continue

            # Convert datapack if not already converted
            convert_datapack(
                loader=RcabenchDatapackLoader(
                    src_folder=input_path, datapack=injection_name
                ),
                dst_folder=converted_input_path,
                skip_finished=True,
            )

            # Process the converted datapack - use ALL samples for testing
            samples = process_datapack_parallel(
                converted_input_path,
                dataset_name,
                sample_interval,
                sample_step,
                metadata,  # Use metadata from training
                log_files,
                metric_files,
                trace_files,
                "drain.ini",
                "cache/drain/temp",
            )

            all_test_samples.extend(samples)
            logger.info(
                f"Processed test datapack {injection_name}: {len(samples)} samples"
            )

        logger.info(
            f"Total test samples from dataset ID {dataset_id}: {len(all_test_samples)}"
        )
        return all_test_samples

    def prepare_train_data_by_dataset_id(
        self,
        config: Config,
        dataset_id: int,
    ) -> Tuple[DataLoader, DataLoader, DatasetMetadata]:
        """
        Prepare training data using dataset ID from RCABench platform
        Split into train and validation sets

        Args:
            config: Configuration object
            dataset_id: Dataset ID from RCABench platform for training

        Returns:
            Tuple of (train_loader, val_loader, metadata)
        """
        # Load training samples from RCABench platform
        train_samples, metadata = self.load_train_data_by_dataset_id(config, dataset_id)

        # Save metadata for later use in testing - 使用统一的metadata目录
        dataset_name = config.get("dataset")
        metadata_dir = Path(config.get("paths.metadata"))
        metadata_dir.mkdir(exist_ok=True)
        metadata_path = (
            metadata_dir / f"{dataset_name}_dataset_{dataset_id}_metadata.pkl"
        )
        metadata.to_pkl(str(metadata_path))
        logger.info(f"Saved training metadata to {metadata_path}")

        # Split training data into train and validation
        from src.eadro.handlers import EadroTrainingHandler

        training_handler = EadroTrainingHandler(
            "cpu"
        )  # Device doesn't matter for splitting

        val_ratio = config.get("training.validation_ratio", 0.1)
        train_loader, val_loader = training_handler.split_train_data_for_validation(
            train_samples, metadata, config, val_ratio
        )

        self.metadata = metadata
        self.train_loader = train_loader
        self.test_loader = val_loader  # Use val_loader as test_loader for consistency

        return train_loader, val_loader, metadata

    def load_metadata_by_train_dataset_id(
        self,
        config: Config,
        train_dataset_id: int,
    ) -> DatasetMetadata:
        """
        Load metadata that was saved during training phase

        Args:
            config: Configuration object
            train_dataset_id: Dataset ID that was used for training

        Returns:
            Loaded metadata
        """
        dataset_name = config.get("dataset")
        metadata_dir = Path(config.get("paths.metadata"))
        metadata_path = (
            metadata_dir / f"{dataset_name}_dataset_{train_dataset_id}_metadata.pkl"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Training metadata not found at {metadata_path}. "
                f"Please ensure you have trained with dataset ID {train_dataset_id} first."
            )

        metadata = DatasetMetadata.from_pkl(str(metadata_path))
        if metadata is None:
            raise ValueError(f"Failed to load metadata from {metadata_path}")

        logger.info(f"Loaded training metadata from {metadata_path}")
        return metadata

    def prepare_test_data_by_dataset_id(
        self,
        config: Config,
        test_dataset_id: int,
        train_dataset_id: int,
    ) -> Tuple[DataLoader, DatasetMetadata]:
        """
        Prepare test data using dataset ID from RCABench platform
        Load metadata from training phase

        Args:
            config: Configuration object
            test_dataset_id: Dataset ID from RCABench platform for testing
            train_dataset_id: Dataset ID that was used for training (for metadata)

        Returns:
            Tuple of (test_loader, metadata)
        """
        # Load metadata from training phase
        metadata = self.load_metadata_by_train_dataset_id(config, train_dataset_id)

        # Load test samples from RCABench platform
        test_samples = self.load_test_data_by_dataset_id(
            config, test_dataset_id, metadata
        )

        # Create test dataset and loader
        test_dataset = ChunkDataset(test_samples, metadata)
        batch_size = config.get("training.batch_size")

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for testing
            collate_fn=collate_fn,
            pin_memory=True,
        )

        self.test_loader = test_loader
        return test_loader, metadata

    def load_single_datapack_for_inference(
        self,
        config: Config,
        datapack_path: Path,
        metadata: Optional[DatasetMetadata] = None,
    ) -> Tuple[List[DataSample], DatasetMetadata]:
        """
        Load data from a single converted datapack for inference

        Args:
            config: Configuration object
            datapack_path: Path to converted datapack
            metadata: Optional pre-loaded metadata

        Returns:
            Tuple of (samples, metadata)
        """
        from src.preprocessing.processor import process_datapack_parallel

        if not datapack_path.exists():
            raise FileNotFoundError(f"Datapack directory not found: {datapack_path}")

        dataset_name = config.get("dataset")
        dataset_config = config.get(f"datasets.{dataset_name}")

        sample_interval = int(dataset_config.get("sample_interval", 30))
        sample_step = int(dataset_config.get("sample_step", 30))
        log_files = dataset_config.get("log_files", [])
        metric_files = dataset_config.get("metric_files", [])
        trace_files = dataset_config.get("trace_files", [])

        # Create metadata if not provided
        if metadata is None:
            from src.preprocessing.processor import Processor

            processor = Processor({}, conf="settings.toml")
            processor.datapacks = [datapack_path]
            processor.derive_files()
            metadata = processor.create_metadata()
            logger.info("Created new metadata from datapack")

        # Process the datapack
        samples = process_datapack_parallel(
            datapack_path,
            dataset_name,
            sample_interval,
            sample_step,
            metadata,
            log_files,
            metric_files,
            trace_files,
            "drain.ini",
            "cache/drain/temp",
        )

        logger.info(f"Generated {len(samples)} samples from {datapack_path}")
        return samples, metadata

    def prepare_data_for_single_inference(
        self,
        config: Config,
        datapack_path: Path,
        metadata: Optional[DatasetMetadata] = None,
    ) -> Tuple[Optional[DataLoader], DataLoader, DatasetMetadata]:
        """
        Prepare data for inference on a single converted datapack

        Args:
            config: Configuration object
            datapack_path: Path to converted datapack
            metadata: Optional pre-loaded metadata

        Returns:
            Tuple of (None, test_loader, metadata)
        """
        samples, metadata = self.load_single_datapack_for_inference(
            config, datapack_path, metadata
        )

        # Create dataset and loader
        dataset = ChunkDataset(samples, metadata)
        batch_size = config.get("training.batch_size")

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        self.metadata = metadata
        self.test_loader = test_loader
        self.train_loader = None

        return None, test_loader, metadata

    def load_data_from_converted_datapacks(
        self,
        config: Config,
        datapack_paths: List[Path],
        metadata: DatasetMetadata,
        label: str = "test",
    ) -> List[DataSample]:
        """
        Load data directly from converted datapack paths (for inference)

        Args:
            config: Configuration object
            datapack_paths: List of converted datapack paths
            metadata: Dataset metadata
            label: Data label (train/test)

        Returns:
            List of data samples
        """
        from src.preprocessing.processor import process_datapack_parallel

        dataset_name = config.get("dataset")
        dataset_config = getattr(config, "datasets", {}).get(dataset_name, {})

        sample_interval = int(dataset_config.get("sample_interval", 30))
        sample_step = int(dataset_config.get("sample_step", 30))

        log_files = dataset_config.get("log_files", [])
        metric_files = dataset_config.get("metric_files", [])
        trace_files = dataset_config.get("trace_files", [])

        all_samples = []

        for datapack_path in datapack_paths:
            logger.info(f"Processing converted datapack: {datapack_path}")

            # Process each datapack
            samples = process_datapack_parallel(
                datapack_path,
                dataset_name,
                sample_interval,
                sample_step,
                metadata,
                log_files,
                metric_files,
                trace_files,
                "drain.ini",
                "cache/drain/temp",
            )

            all_samples.extend(samples)
            logger.info(f"Loaded {len(samples)} samples from {datapack_path}")

        logger.info(f"Total samples loaded: {len(all_samples)}")
        return all_samples

    def prepare_data_for_inference(
        self,
        config: Config,
        datapack_paths: List[Path],
        metadata: Optional[DatasetMetadata] = None,
    ) -> Tuple[DataLoader, DataLoader, DatasetMetadata]:
        """
        Prepare data for inference using converted datapack paths

        Args:
            config: Configuration object
            datapack_paths: List of converted datapack paths
            metadata: Optional pre-loaded metadata

        Returns:
            Tuple of (train_loader, test_loader, metadata) - train_loader will be None
        """
        if metadata is None:
            # Try to load metadata from metadata directory
            dataset_name = config.get("dataset")
            metadata_dir = Path(config.get("paths.metadata"))
            metadata_path = metadata_dir / f"{dataset_name}_metadata.pkl"

            if metadata_path.exists():
                metadata = DatasetMetadata.from_pkl(str(metadata_path))
                logger.info(f"Loaded metadata from {metadata_path}")
            else:
                # Create metadata from the datapacks
                from src.preprocessing.processor import Processor

                processor = Processor({}, conf="settings.toml")
                processor.datapacks = list(datapack_paths)
                processor.derive_files()
                metadata = processor.create_metadata()
                logger.info("Created new metadata from datapacks")

        # Load samples from converted datapacks
        samples = self.load_data_from_converted_datapacks(
            config, datapack_paths, metadata, "test"
        )

        # Create dataset and loader
        dataset = ChunkDataset(samples, metadata)
        batch_size = config.get("training.batch_size")

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        self.metadata = metadata
        self.test_loader = test_loader
        # No training data for inference
        self.train_loader = None

        return None, test_loader, metadata

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


class OptimizerAdapter:
    """Adapter to make torch optimizers compatible with HasParamGroups protocol"""

    def __init__(self, optimizer: torch.optim.Optimizer):  # type: ignore
        self.optimizer = optimizer
        # Directly expose param_groups as an attribute for protocol compatibility
        self.param_groups = optimizer.param_groups

    def state_dict(self) -> Dict[str, torch.Tensor]:
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


class EadroModelHandler(ModelHandler[MainModel]):
    def __init__(self, device: str):
        self.device = device

    def save_model(self, model: MainModel, path: str) -> None:
        torch.save(model.state_dict(), path)

    def load_model(self, path: str) -> MainModel:
        raise NotImplementedError(
            "Model loading requires separate architecture creation"
        )

    def get_model_state(self, model: MainModel) -> Dict[str, torch.Tensor]:
        return model.state_dict()

    def load_model_state(
        self, model: MainModel, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        model.load_state_dict(state_dict)


class EadroTrainingHandler(
    TrainingHandler[
        MainModel, Tuple[DataLoader, DataLoader, DatasetMetadata], OptimizerAdapter
    ]
):
    """Training handler for EADRO model"""

    def __init__(self, device: str):
        self.device = device
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def split_train_data_for_validation(
        self,
        train_samples: List[DataSample],
        metadata: DatasetMetadata,
        config: Config,
        val_ratio: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Split training data into train and validation sets

        Args:
            train_samples: All training samples
            metadata: Dataset metadata
            config: Configuration object
            val_ratio: Ratio of data to use for validation

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Shuffle samples before splitting
        random.shuffle(train_samples)

        # Split data
        val_size = int(len(train_samples) * val_ratio)
        val_samples = train_samples[:val_size]
        train_samples = train_samples[val_size:]

        logger.info(
            f"Split data: {len(train_samples)} train samples, {len(val_samples)} validation samples"
        )

        # Create datasets and loaders
        train_dataset = ChunkDataset(train_samples, metadata)
        val_dataset = ChunkDataset(val_samples, metadata)

        batch_size = config.get("training.batch_size")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train_epoch(
        self,
        model: MainModel,
        data: Tuple[DataLoader, DataLoader, DatasetMetadata],
        optimizer: OptimizerAdapter,
        **kwargs,
    ) -> MetricsDict:
        """Train model for one epoch"""
        train_loader, _, _ = data
        model.train()

        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (graph, ground_truth) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            result = model.forward(graph.to(self.device), ground_truth.to(self.device))
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
        model: MainModel,
        data: Tuple[DataLoader, DataLoader, DatasetMetadata],
        **kwargs,
    ) -> MetricsDict:
        """Validate model and return metrics including validation loss"""
        _, val_loader, _ = data
        model.eval()
        hrs, ndcgs = np.zeros(5), np.zeros(5)
        TP, FP, FN = 0, 0, 0
        batch_cnt, epoch_loss = 0, 0.0

        with torch.no_grad():
            for graph, ground_truths in val_loader:
                res = model.forward(
                    graph.to(self.device), ground_truths.to(self.device)
                )
                for idx, faulty_nodes in enumerate(res["y_pred"]):
                    culprit = ground_truths[idx].item()
                    if culprit == -1:
                        if faulty_nodes[0] == -1:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if faulty_nodes[0] == -1:
                            FN += 1
                        else:
                            TP += 1
                            rank = list(faulty_nodes).index(culprit)
                            for j in range(5):
                                hrs[j] += int(rank <= j)  # Calculate Hit Rate
                                ndcgs[j] += ndcg_score(
                                    np.array([res["y_prob"][idx]]).reshape(1, -1),
                                    np.array([res["pred_prob"][idx]]).reshape(1, -1),
                                    k=j + 1,
                                )
                epoch_loss += res["loss"].item()
                batch_cnt += 1

        avg_val_loss = epoch_loss / batch_cnt if batch_cnt > 0 else 0.0
        pos = TP + FN  # Total number of positive samples

        eval_results = {
            "val_loss": avg_val_loss,
            "F1": TP * 2.0 / (TP + FP + pos) if (TP + FP + pos) > 0 else 0,
            "Rec": TP * 1.0 / pos if pos > 0 else 0,  # Recall
            "Pre": TP * 1.0 / (TP + FP) if (TP + FP) > 0 else 0,  # Precision
        }

        # Calculate Hit Rate and NDCG metrics
        for j in [1, 3, 5]:
            eval_results["HR@" + str(j)] = hrs[j - 1] * 1.0 / pos if pos > 0 else 0
            eval_results["ndcg@" + str(j)] = ndcgs[j - 1] * 1.0 / pos if pos > 0 else 0

        return eval_results

    def should_stop_early(self, val_metrics: MetricsDict, config: Config) -> bool:
        """
        Check if training should stop early based on validation loss

        Args:
            val_metrics: Validation metrics including val_loss
            config: Configuration object

        Returns:
            True if training should stop
        """
        patience = config.get("training.patience", 10)
        min_delta = config.get("training.min_delta", 1e-4)

        current_val_loss = val_metrics.get("val_loss", float("inf"))

        # Check if validation loss improved
        if current_val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            logger.info(f"Validation loss improved to {current_val_loss:.6f}")
            return False
        else:
            self.patience_counter += 1
            logger.info(
                f"Validation loss did not improve. Patience: {self.patience_counter}/{patience}"
            )

            if self.patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered. Best validation loss: {self.best_val_loss:.6f}"
                )
                return True

            return False

    def setup_optimizer(self, model: MainModel, config: Config) -> OptimizerAdapter:
        """Setup optimizer for training"""
        lr = config.get("training.lr")
        adam_optimizer = Adam(model.parameters(), lr=lr)
        return OptimizerAdapter(adam_optimizer)


class EadroInferenceHandler(
    InferenceHandler[MainModel, Tuple[DataLoader, DataLoader, DatasetMetadata]]
):
    def __init__(self, device: str):
        self.device = device

    def predict(
        self,
        model: MainModel,
        data: Tuple[DataLoader, DataLoader, DatasetMetadata],
    ) -> Dict:
        # Use test_loader if available, otherwise use train_loader for inference
        train_loader, test_loader, metadata = data
        inference_loader = test_loader if test_loader is not None else train_loader

        if inference_loader is None:
            raise ValueError("No data loader available for inference")

        model.eval()

        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for graph, ground_truths in inference_loader:
                result = model.forward(
                    graph.to(self.device), ground_truths.to(self.device)
                )
                predictions = result["y_pred"]
                probabilities = result["pred_prob"]

                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)

        return {
            "predictions": all_predictions,
            "probabilities": all_probabilities,
            "metadata": metadata,
        }

    def predict_from_datapacks(
        self,
        model: MainModel,
        config: Config,
        datapack_paths: List[Path],
        metadata: Optional[DatasetMetadata] = None,
    ) -> Dict:
        """
        Direct prediction from converted datapack paths

        Args:
            model: Trained model
            config: Configuration object
            datapack_paths: List of converted datapack paths
            metadata: Optional pre-loaded metadata

        Returns:
            Dictionary containing predictions and probabilities
        """
        from src.eadro.handlers import EadroDataHandler

        data_handler = EadroDataHandler()

        # Prepare data for inference
        _, test_loader, metadata = data_handler.prepare_data_for_inference(
            config, datapack_paths, metadata
        )

        # Run prediction
        data = (None, test_loader, metadata)
        return self.predict(model, data)

    def postprocess_results(self, predictions: Dict, **kwargs) -> PredictionResult:
        predictions_list = predictions["predictions"]
        probabilities_list = predictions["probabilities"]
        metadata = predictions["metadata"]

        service_scores = defaultdict(float)

        position_weights = [1.0, 0.8, 0.6, 0.4, 0.2]

        for pred_list in predictions_list:
            for position, service_id in enumerate(pred_list):
                if position < len(position_weights):
                    service_scores[service_id] += position_weights[position]

        sorted_services = sorted(
            service_scores.items(), key=lambda x: x[1], reverse=True
        )
        final_service_list = [service_id for service_id, score in sorted_services[:5]]

        if len(final_service_list) < 5:
            all_service_ids = set()
            for pred_list in predictions_list:
                all_service_ids.update(pred_list)

            remaining_services = list(all_service_ids - set(final_service_list))
            final_service_list.extend(remaining_services[: 5 - len(final_service_list)])

        service_names = [
            metadata.services[service_id].name for service_id in final_service_list
        ]
        return {
            "predictions": service_names,
        }

    def predict_from_single_datapack(
        self,
        model: MainModel,
        config: Config,
        datapack_path: Path,
        metadata: Optional[DatasetMetadata] = None,
    ) -> Dict:
        """
        Direct prediction from a single converted datapack path

        Args:
            model: Trained model
            config: Configuration object
            datapack_path: Path to converted datapack
            metadata: Optional pre-loaded metadata

        Returns:
            Dictionary containing predictions and probabilities
        """
        data_handler = EadroDataHandler()

        # Prepare data for inference
        _, test_loader, metadata = data_handler.prepare_data_for_single_inference(
            config, datapack_path, metadata
        )

        # Run prediction
        data = (None, test_loader, metadata)
        return self.predict(model, data)


def create_eadro_model(
    event_num: int, metric_num: int, node_num: int, device: str, config: Config
) -> MainModel:
    model = MainModel(
        event_num=event_num,
        metric_num=metric_num,
        node_num=node_num,
        device=device,
        config=config,
    )
    return model.to(device)
    return model.to(device)
