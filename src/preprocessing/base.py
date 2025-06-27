from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from dynaconf import Dynaconf
import pickle
import os
import json
import numpy as np


@dataclass
class ServiceMetadata:
    id: int
    name: str


@dataclass
class MetricMetadata:
    name: str
    min_value: Optional[float] = None  # unit: milliseconds
    max_value: Optional[float] = None  # unit: milliseconds


@dataclass
class LogTemplateMetadata:
    template: str
    id: int
    frequency: int = 0


@dataclass
class TraceMetadata:
    span_name: str
    min_duration: Optional[float] = None  # unit: milliseconds
    max_duration: Optional[float] = None  # unit: milliseconds


@dataclass
class FaultMetadata:
    fault_types: List[str] = field(default_factory=list)


@dataclass
class DatasetMetadata:
    """
    Metadata for a dataset (which contains a series of datapacks), including services mapping, metric recording, log templates, and fault information.
    """

    dataset_name: str

    services: List[ServiceMetadata] = field(default_factory=list)
    service_name_to_id: Dict[str, int] = field(default_factory=dict)

    metrics: List[MetricMetadata] = field(default_factory=list)
    metric_names: List[str] = field(default_factory=list)
    metric_name_to_id: Dict[str, int] = field(default_factory=dict)

    log_templates: List[LogTemplateMetadata] = field(default_factory=list)
    log_template_to_id: Dict[str, int] = field(default_factory=dict)

    traces: List[TraceMetadata] = field(default_factory=list)
    service_calling_edges: List[List[int]] = field(default_factory=list)

    fault_info: FaultMetadata = field(default_factory=FaultMetadata)

    def to_pkl(self, path: str):
        if os.path.exists(path):
            print(f"Warning: Overwriting existing metadata file at {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)  # 直接保存 self 而不是 asdict(self)

    @staticmethod
    def from_pkl(path: str) -> "DatasetMetadata | None":
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading DatasetMetadata from {path}: {e}")
            return None

    def to_json(self, path: str):
        if os.path.exists(path):
            print(f"Warning: Overwriting existing metadata file at {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)


@dataclass
class DataSample:
    # a minial data sample, which can be used to represent a single data point in the dataset, which will be used for training/testing
    abnormal: bool
    gt_service: str = ""
    fault_type: str = ""

    log: np.ndarray = field(default_factory=lambda: np.array([]))
    trace: np.ndarray = field(default_factory=lambda: np.array([]))
    metric: np.ndarray = field(default_factory=lambda: np.array([]))


class BaseParser(ABC):
    @abstractmethod
    def parse(self, *args, **kwargs) -> Any:
        pass


@dataclass
class BaseConfig:
    @abstractmethod
    def load(self, config_path: str):
        pass


class DataProcessor:
    @abstractmethod
    def _load_config(self, config_path: str) -> Dynaconf:
        pass

    @abstractmethod
    def process_dataset(self):
        """
        1. create metadata: create_metadata
        2. process each datapack: process_datapack
           2.1 process log
           2.2 process metrics
           2.3 process traces
           2.4 load ground truth (labels)
        3. split to chunks(each chunk is the data within a certain time range, will be used for training/testing)
        """

    @abstractmethod
    def create_metadata(self) -> DatasetMetadata:
        pass

    @abstractmethod
    def process_datapack(self, datapack: str):
        pass


@dataclass
class ProcessingResult:
    metadata: DatasetMetadata
    chunks_path: str
    train_chunks_path: str
    test_chunks_path: str
    total_chunks: int
    total_datapacks: int
