from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dynaconf import Dynaconf
import pickle
import os
import json
import numpy as np
from datetime import datetime, timedelta


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

    # Add computed property for service ID (similar to old preprocessing culprit field)
    def get_gt_service_id(self, service_name_to_id: dict) -> int:
        """Get ground truth service ID, return -1 if not found or no fault"""
        if not self.abnormal or not self.gt_service:
            return -1
        return service_name_to_id.get(self.gt_service, -1)


@dataclass
class TimeSeriesDataSample:
    """连续时间序列数据样本 - 改造后的数据结构"""

    datapack_name: str

    # 时间信息
    start_time: datetime
    end_time: datetime
    time_resolution: int  # 秒，时间分辨率

    # 标签信息 - 正常和异常时间段
    normal_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)
    abnormal_periods: List[Tuple[datetime, datetime, str, str]] = field(
        default_factory=list
    )  # start, end, gt_service, fault_type

    # 时间序列数据 (time_steps, services, features)
    log_series: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # (time_steps, services, log_templates)
    metric_series: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # (time_steps, services, metrics)
    trace_series: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # (time_steps, services, trace_features)

    def get_time_steps(self) -> int:
        """获取时间步数"""
        return int(
            (self.end_time - self.start_time).total_seconds() / self.time_resolution
        )

    def get_time_window(self, start_idx: int, window_size: int) -> "DataSample":
        """提取指定时间窗口的数据样本"""
        end_idx = start_idx + window_size

        # 确保索引在有效范围内
        if end_idx > len(self.log_series):
            raise IndexError(
                f"Window end index {end_idx} exceeds data length {len(self.log_series)}"
            )

        # 计算窗口对应的实际时间
        window_start_time = self.start_time + timedelta(
            seconds=start_idx * self.time_resolution
        )
        window_end_time = self.start_time + timedelta(
            seconds=end_idx * self.time_resolution
        )

        # 获取窗口的标签信息
        abnormal, gt_service, fault_type = self.get_labels_for_window(
            window_start_time, window_end_time
        )

        # 提取窗口数据
        window_log = (
            self.log_series[start_idx:end_idx]
            if self.log_series.size > 0
            else np.array([])
        )
        window_metric = (
            self.metric_series[start_idx:end_idx]
            if self.metric_series.size > 0
            else np.array([])
        )
        window_trace = (
            self.trace_series[start_idx:end_idx]
            if self.trace_series.size > 0
            else np.array([])
        )

        # 对于log数据，需要聚合时间维度（求和）
        if window_log.size > 0:
            aggregated_log = np.sum(window_log, axis=0)  # 聚合时间维度
        else:
            aggregated_log = np.array([])

        # 对于metric和trace数据，保持原有的3D结构 (services, time, features)
        if window_metric.size > 0:
            # 转置为 (services, time, features) 格式
            aggregated_metric = np.transpose(window_metric, (1, 0, 2))
        else:
            aggregated_metric = np.array([])

        if window_trace.size > 0:
            # 转置为 (services, time, features) 格式
            aggregated_trace = np.transpose(window_trace, (1, 0, 2))
        else:
            aggregated_trace = np.array([])

        return DataSample(
            abnormal=abnormal,
            gt_service=gt_service,
            fault_type=fault_type,
            log=aggregated_log,
            metric=aggregated_metric,
            trace=aggregated_trace,
        )

    def get_labels_for_window(
        self, start_time: datetime, end_time: datetime
    ) -> Tuple[bool, str, str]:
        """获取时间窗口的标签信息"""
        # 检查窗口是否与异常时间段重叠
        for (
            abnormal_start,
            abnormal_end,
            gt_service,
            fault_type,
        ) in self.abnormal_periods:
            # 检查时间重叠：窗口开始时间 < 异常结束时间 且 窗口结束时间 > 异常开始时间
            if start_time < abnormal_end and end_time > abnormal_start:
                return True, gt_service, fault_type

        # 如果没有异常重叠，返回正常状态
        return False, "", ""


@dataclass
class TimeSeriesDatapack:
    """包含多个TimeSeriesDataSample的数据包"""

    samples: List[TimeSeriesDataSample] = field(default_factory=list)
    metadata: Optional[DatasetMetadata] = None

    def save(self, path: str):
        """保存到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> Optional["TimeSeriesDatapack"]:
        """从文件加载"""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading TimeSeriesDatapack from {path}: {e}")
            return None


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
