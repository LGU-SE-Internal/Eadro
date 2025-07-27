from .base import (
    DataProcessor,
    DatasetMetadata,
    BaseParser,
    ServiceMetadata,
    MetricMetadata,
    LogTemplateMetadata,
    DataSample,
    TraceMetadata,
)
from dynaconf import Dynaconf
from pathlib import Path
import polars as pl
from tqdm import tqdm
from .log import DrainProcessor
from datetime import datetime
import numpy as np
import pickle
from loguru import logger
from rcabench.openapi import InjectionApi, ApiClient, Configuration
import random
import time
from .utils import fmap_processpool
import functools
from .log import preaggregate_logs
from .trace import preaggregate_traces
from .metric import preaggregate_metrics
from .label import extract_eadro_labels, extract_rcabench_labels
from ..exp.config import Config


def process_datapack_parallel(
    datapack: Path,
    dataset: str,
    sample_interval: int,
    sample_step: int,
    metadata: DatasetMetadata,
    log_files: list[str],
    metric_files: list[str],
    trace_files: list[str],
    drain_config_path: str = "drain.ini",
    drain_save_path: str = "cache/drain/temp",
) -> list[DataSample]:
    config = Dynaconf(settings_files=["settings.toml"])
    dataset_config = getattr(config, "datasets", {}).get(dataset, {})  # type: ignore

    samples = []

    if dataset == "rcabench":
        try:
            (
                normal_st,
                normal_et,
                abnormal_st,
                abnormal_et,
                gt_service,
                fault_type,
            ) = extract_rcabench_labels(datapack, dataset_config)
        except Exception as e:
            logger.error(f"Failed to extract labels for {datapack}: {e}")
            return []
    elif dataset == "sn" or dataset == "tt":
        normal_st, normal_et, abnormal_st, abnormal_et, gt_service, fault_type = (
            extract_eadro_labels(datapack, dataset_config)
        )

    interval = sample_interval
    sample_step = sample_step
    assert isinstance(sample_step, int), "Sample step should be an integer."
    assert interval is not None, (
        f"Sample interval for {dataset} is not defined in the config."
    )
    assert isinstance(interval, int), "Sample interval should be an integer."

    time_range = []
    if normal_st != datetime(1970, 1, 1, 0, 0, 0) and normal_et != datetime(
        1970, 1, 1, 0, 0, 0
    ):
        time_range.append((normal_st, normal_et, "", ""))
    if abnormal_st != datetime(1970, 1, 1, 0, 0, 0) and abnormal_et != datetime(
        1970, 1, 1, 0, 0, 0
    ):
        time_range.append((abnormal_st, abnormal_et, fault_type, gt_service))

    def concat_and_cache_dfs(
        files: list[str], start: datetime, end: datetime
    ) -> pl.DataFrame:
        lazy_frames = [
            pl.scan_parquet(datapack / f).filter(
                (pl.col("time") >= start) & (pl.col("time") <= end)
            )
            for f in files
        ]
        if not lazy_frames:
            return pl.DataFrame()
        return pl.concat(lazy_frames).collect()

    for start, end, fault_type, gt_service in time_range:
        # Cache all data for the current time range (normal/abnormal)
        log_df_cached = concat_and_cache_dfs(log_files, start, end)
        metric_df_cached = concat_and_cache_dfs(metric_files, start, end)
        trace_df_cached = concat_and_cache_dfs(trace_files, start, end)
        preaggregated_metrics = preaggregate_metrics(
            metric_df_cached, start, end, metadata
        )
        preaggregated_traces = preaggregate_traces(
            trace_df_cached, start, end, metadata
        )
        preaggregated_logs = preaggregate_logs(
            log_df_cached,
            start,
            end,
            metadata,
            drain_config_path,
            drain_save_path,
        )

        total_seconds = int((end - start).total_seconds())

        for i in range(0, total_seconds - interval + 1, sample_step):
            if i + interval > total_seconds:
                break

            sample = DataSample(
                abnormal=gt_service != "",
                gt_service=gt_service,
                fault_type=fault_type,
            )

            # Direct slicing from pre-aggregated arrays (no repeated computation)
            sample.metric = preaggregated_metrics[:, i : i + interval, :]
            sample.trace = preaggregated_traces[:, i : i + interval, :]

            # For logs, sum counts over the time window
            log_counts_window = preaggregated_logs[:, i : i + interval, :]
            sample.log = np.sum(log_counts_window, axis=1)  # Sum along time axis

            # Validation
            assert sample.metric.shape == (
                len(metadata.services),
                interval,
                len(metadata.metric_names),
            )
            assert sample.trace.shape == (
                len(metadata.services),
                interval,
                2,
            )
            assert sample.log.shape == (
                len(metadata.services),
                len(metadata.log_templates) + 1,
            )

            samples.append(sample)

    return samples


class Processor(DataProcessor):
    def __init__(self, parsers: dict[str, BaseParser], conf: str = "config.toml"):
        self.config = self._load_config(conf)
        self.dataset: str = self.config.get("dataset")  # type: ignore
        assert self.dataset != "", "Dataset name cannot be empty."
        self.dataset_config = self.config.get("datasets")[self.dataset]  # type: ignore

        self.parsers: dict[str, BaseParser] = parsers
        self.drain = DrainProcessor(conf="drain.ini", save_path="cache/drain/temp")

        self._file_cache = {}

        self.datapack_dir = self.dataset_config.root_path  # type: ignore
        # assert isinstance(self.datapack_dir, str)
        # assert self.datapack_dir != ""
        self.datapack_path = Path(self.datapack_dir)
        # assert self.datapack_path.exists(), (
        #     f"Datapack path {self.datapack_path} does not exist."
        # )
        # assert self.datapack_path.is_dir()

        pl.Config.set_streaming_chunk_size(50000)
        pl.Config.set_tbl_rows(20)

    def _load_config(self, config_path: str) -> Config:
        assert config_path != "", "Config path cannot be empty."
        return Config(config_file=Path(config_path))

    def _validate_datapack(self, pack: Path) -> bool:
        if not (pack.exists() and pack.is_dir()):
            logger.warning(
                f"Datapack {pack} does not exist or is not a directory. Skipping."
            )
            return False

        missing_files = []

        for metric_file in self.metric_files:
            if not (pack / metric_file).exists():
                missing_files.append(f"metric file: {metric_file}")

        for log_file in self.log_files:
            if not (pack / log_file).exists():
                missing_files.append(f"log file: {log_file}")

        for trace_file in self.trace_files:
            if not (pack / trace_file).exists():
                missing_files.append(f"trace file: {trace_file}")

        if missing_files:
            logger.warning(
                f"Datapack {pack} is missing files: {', '.join(missing_files)}. Skipping."
            )
            return False

        return True

    def _load_datapacks(self) -> list[Path]:
        all_dirs = [dp for dp in self.datapack_path.iterdir() if dp.is_dir()]
        valid_packs = []
        for pack in all_dirs:
            if self._validate_datapack(pack):
                valid_packs.append(pack)

        if not valid_packs:
            raise ValueError("No valid datapacks found after validation.")

        return valid_packs

    def _load_datapacks_from_folder(self, dataset_folder: str) -> list[Path]:
        """
        从指定的数据集文件夹加载数据包
        
        Args:
            dataset_folder: 数据集文件夹名称（如 __dev__rcabench_test_r1）
        
        Returns:
            数据包路径列表
        """
        folder_path = self.datapack_path / dataset_folder
        
        if not folder_path.exists():
            raise ValueError(f"Dataset folder '{dataset_folder}' not found at {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"'{dataset_folder}' is not a directory")
        
        # 收集该文件夹下的所有案例目录
        case_dirs = [case_dp for case_dp in folder_path.iterdir() if case_dp.is_dir()]
        
        if not case_dirs:
            raise ValueError(f"No case directories found in '{dataset_folder}'")
        
        valid_packs = []
        for pack in case_dirs:
            if self._validate_datapack(pack):
                valid_packs.append(pack)

        if not valid_packs:
            raise ValueError(f"No valid datapacks found in '{dataset_folder}' after validation.")

        logger.info(
            f"Loaded {len(valid_packs)} valid datapacks from folder '{dataset_folder}' out of {len(case_dirs)} total."
        )
        return valid_packs

    def process_dataset(self, use_parallel: bool = False, n_workers: int | None = None, split: str = "train"):
        """
        处理数据集
        
        Args:
            use_parallel: 是否使用并行处理
            n_workers: 工作进程数量
            split: 数据集分割类型，可以是 "train" 或 "test"
        """
        self.derive_files()
        self.datapacks = self._load_datapacks()
        self.metadata = self.create_metadata()

        if use_parallel and len(self.datapacks) > 1:
            # Parallel processing of datapacks using fmap_processpool
            if n_workers is None:
                n_workers = 16

            logger.info(
                f"Processing {len(self.datapacks)} {split} datapacks in parallel using {n_workers} workers with fmap_processpool"
            )

            # Type ignore to handle Dynaconf typing issues
            sample_interval = int(self.dataset_config.sample_interval)  # type: ignore
            sample_step = int(self.dataset_config.sample_step)  # type: ignore

            tasks = [
                functools.partial(
                    process_datapack_parallel,
                    dp,
                    self.dataset,
                    sample_interval,
                    sample_step,
                    self.metadata,
                    self.log_files,
                    self.metric_files,
                    self.trace_files,
                    "drain.ini",
                    "cache/drain/temp",
                )
                for dp in self.datapacks
            ]

            for i in range(0, len(tasks), n_workers):
                batch_tasks = tasks[i : i + n_workers]
                logger.info(
                    f"Processing {split} batch {i // n_workers + 1}/{(len(tasks) + n_workers - 1) // n_workers} (datapacks {i + 1}-{min(i + n_workers, len(tasks))})"
                )

                batch_results = fmap_processpool(
                    batch_tasks, parallel=n_workers, ignore_exceptions=False
                )

                batch_samples = []
                for datapack_samples in batch_results:
                    batch_samples.extend(datapack_samples)

                batch_file = (
                    f".cache/{self.dataset}_{split}_samples_batch_{i // n_workers + 1}.pkl"
                )
                Path(".cache").mkdir(exist_ok=True)
                with open(batch_file, "wb") as f:
                    pickle.dump(batch_samples, f)
                logger.info(
                    f"Saved {split} batch {i // n_workers + 1} with {len(batch_samples)} samples to {batch_file}"
                )

                del batch_samples
                del batch_results

    def process_dataset_folder(self, dataset_folder: str, label: str, use_parallel: bool = False, n_workers: int | None = None):
        """
        处理指定文件夹的数据集
        
        Args:
            dataset_folder: 数据集文件夹名称（如 __dev__rcabench_test_r1）
            label: 数据集标签（train 或 test）
            use_parallel: 是否使用并行处理
            n_workers: 工作进程数量
        """
        assert label in ["train", "test"], f"Label must be 'train' or 'test', got {label}"
        
        logger.info(f"Processing dataset folder: {dataset_folder} with label: {label}")
        
        self.derive_files()
        self.datapacks = self._load_datapacks_from_folder(dataset_folder)
        
        self.metadata = self.create_metadata_for_folder(dataset_folder, label)

        if use_parallel and len(self.datapacks) > 1:
            # Parallel processing of datapacks using fmap_processpool
            if n_workers is None:
                n_workers = 16

            logger.info(
                f"Processing {len(self.datapacks)} datapacks from folder '{dataset_folder}' in parallel using {n_workers} workers with fmap_processpool"
            )

            # Type ignore to handle Dynaconf typing issues
            sample_interval = int(self.dataset_config.sample_interval)  # type: ignore
            sample_step = int(self.dataset_config.sample_step)  # type: ignore

            tasks = [
                functools.partial(
                    process_datapack_parallel,
                    dp,
                    self.dataset,
                    sample_interval,
                    sample_step,
                    self.metadata,
                    self.log_files,
                    self.metric_files,
                    self.trace_files,
                    "drain.ini",
                    "cache/drain/temp",
                )
                for dp in self.datapacks
            ]

            for i in range(0, len(tasks), n_workers):
                batch_tasks = tasks[i : i + n_workers]
                logger.info(
                    f"Processing batch {i // n_workers + 1}/{(len(tasks) + n_workers - 1) // n_workers} (datapacks {i + 1}-{min(i + n_workers, len(tasks))}) for folder '{dataset_folder}'"
                )

                batch_results = fmap_processpool(
                    batch_tasks, parallel=n_workers, ignore_exceptions=False
                )

                batch_samples = []
                for datapack_samples in batch_results:
                    batch_samples.extend(datapack_samples)

                # 使用包含数据集文件夹名称的文件名，避免覆盖
                batch_file = (
                    f".cache/{self.dataset}_{dataset_folder}_{label}_samples_batch_{i // n_workers + 1}.pkl"
                )
                Path(".cache").mkdir(exist_ok=True)
                with open(batch_file, "wb") as f:
                    pickle.dump(batch_samples, f)
                logger.info(
                    f"Saved batch {i // n_workers + 1} with {len(batch_samples)} samples to {batch_file}"
                )

                del batch_samples
                del batch_results
                

    def derive_files(self):
        def load(f) -> list[str]:
            files = self.dataset_config[f]  # type: ignore
            assert files is not None, (
                f"Files for {self.dataset} are not defined in the config."
            )
            assert isinstance(files, list), "Files should be a list."
            assert len(files) > 0, "No files defined for datapack in the config."
            return files

        self.log_files = load("log_files")
        self.metric_files = load("metric_files")
        self.trace_files = load("trace_files")

    def create_metadata(self) -> DatasetMetadata:
        metadata = DatasetMetadata.from_pkl(
            f"{self.config.get('paths.metadata')}/{self.dataset}_metadata.pkl"
        )
        if metadata is None:
            metadata = DatasetMetadata(
                dataset_name=self.dataset,
            )
        else:
            return metadata

        all_metrics = set()
        all_services = set()
        all_span_names = set()
        all_log_templates = set()
        service_calling_edges = []
        metric_stats = {}
        span_duration_stats = {}  # {(span_name, service_name): {"min": float, "max": float}}

        for datapack in tqdm(self.datapacks, desc="Processing datapack metadata"):
            # Batch process all file types to reduce disk I/O
            for metric_file in self.metric_files:
                # Use lazy scan and only select required columns
                df = pl.scan_parquet(datapack / metric_file)
                df = df.select(["metric", "service_name", "value"])

                # Use single aggregation to get all statistics
                stats_and_unique = (
                    df.group_by(["metric", "service_name"])
                    .agg(
                        [
                            pl.col("value").min().alias("min_value"),
                            pl.col("value").max().alias("max_value"),
                        ]
                    )
                    .collect()
                )

                # Extract unique values
                unique_metrics_services = stats_and_unique.select(
                    [pl.col("metric"), pl.col("service_name")]
                ).unique()

                all_metrics.update(unique_metrics_services["metric"].to_list())
                all_services.update(unique_metrics_services["service_name"].to_list())

                # Update metric statistics
                metric_level_stats = stats_and_unique.group_by("metric").agg(
                    [
                        pl.col("min_value").min().alias("global_min"),
                        pl.col("max_value").max().alias("global_max"),
                    ]
                )

                for row in metric_level_stats.iter_rows(named=True):
                    metric_name = row["metric"]
                    min_val = row["global_min"]
                    max_val = row["global_max"]

                    if metric_name not in metric_stats:
                        metric_stats[metric_name] = {"min": min_val, "max": max_val}
                    else:
                        metric_stats[metric_name]["min"] = min(
                            metric_stats[metric_name]["min"], min_val
                        )
                        metric_stats[metric_name]["max"] = max(
                            metric_stats[metric_name]["max"], max_val
                        )

            for log_file in self.log_files:
                # Only select required columns
                df = pl.scan_parquet(datapack / log_file)
                df = df.select(["service_name", "message"])

                unique_services = df.select(pl.col("service_name")).unique().collect()
                all_services.update(unique_services["service_name"].to_list())

                # Batch get unique messages
                logs = (
                    df.select(pl.col("message")).unique().collect()["message"].to_list()
                )
                all_log_templates.update(
                    self.drain.process_batch(random.sample(logs, min(len(logs), 10000)))
                )
                logger.info(f"current log templates: {len(all_log_templates)}")

            for trace_file in self.trace_files:
                df = pl.scan_parquet(datapack / trace_file)

                # Get all required unique values in one go
                unique_data = (
                    df.select(
                        ["service_name", "span_name", "span_id", "parent_span_id"]
                    )
                    .unique()
                    .collect()
                )

                all_services.update(unique_data["service_name"].to_list())
                all_span_names.update(unique_data["span_name"].to_list())

                # Collect duration statistics for each span-service combination
                duration_stats = (
                    df.select(["service_name", "span_name", "duration"])
                    .group_by(["service_name", "span_name"])
                    .agg(
                        [
                            pl.col("duration").min().alias("min_duration"),
                            pl.col("duration").max().alias("max_duration"),
                        ]
                    )
                    .collect()
                )

                for row in duration_stats.iter_rows(named=True):
                    service_name = row["service_name"]
                    span_name = row["span_name"]
                    min_duration = row["min_duration"]
                    max_duration = row["max_duration"]

                    key = (span_name, service_name)
                    if key not in span_duration_stats:
                        span_duration_stats[key] = {
                            "min": min_duration,
                            "max": max_duration,
                        }
                    else:
                        span_duration_stats[key]["min"] = min(
                            span_duration_stats[key]["min"], min_duration
                        )
                        span_duration_stats[key]["max"] = max(
                            span_duration_stats[key]["max"], max_duration
                        )

            # Process service calling edges - only process once
            for trace_file in self.trace_files:
                df = pl.scan_parquet(datapack / trace_file)
                assert (
                    "span_id" in df.collect_schema().names()
                    and "parent_span_id" in df.collect_schema().names()
                ), (
                    f"Trace data in {trace_file} does not contain 'span_id' or 'parent_span_id' columns."
                )

                # Only select required columns
                spans_df = df.select(
                    [
                        pl.col("span_id"),
                        pl.col("parent_span_id"),
                        pl.col("service_name"),
                        pl.col("duration"),
                    ]
                ).collect()

                # Use more efficient join operations
                edges_df = spans_df.join(
                    spans_df.select(
                        [
                            pl.col("span_id").alias("parent_id"),
                            pl.col("service_name").alias("parent_service"),
                        ]
                    ),
                    left_on="parent_span_id",
                    right_on="parent_id",
                    how="inner",
                )

                calling_edges = edges_df.select(
                    [
                        pl.col("parent_service"),
                        pl.col("service_name").alias("child_service"),
                    ]
                ).unique()

                # Use sets to avoid duplicate checking
                existing_edges = set(tuple(edge) for edge in service_calling_edges)

                for row in calling_edges.iter_rows(named=True):
                    parent_service = row["parent_service"]
                    child_service = row["child_service"]
                    if parent_service != child_service:
                        edge = (parent_service, child_service)
                        if edge not in existing_edges:
                            service_calling_edges.append(
                                [parent_service, child_service]
                            )
                            existing_edges.add(edge)

        metadata.services = [
            ServiceMetadata(name=service, id=i)
            for i, service in enumerate(sorted(all_services))
        ]
        metadata.service_name_to_id = {
            service.name: service.id for service in metadata.services
        }
        metadata.metric_names = sorted(all_metrics)
        metadata.metrics = [
            MetricMetadata(
                name=metric,
                min_value=metric_stats[metric]["min"],
                max_value=metric_stats[metric]["max"],
            )
            for metric in metadata.metric_names
        ]
        metadata.metric_name_to_id = {
            metric: i for i, metric in enumerate(metadata.metric_names)
        }

        metadata.log_templates = [
            LogTemplateMetadata(template=template, id=i)
            for i, template in enumerate(sorted(all_log_templates))
        ]
        metadata.log_template_to_id = {
            template.template: template.id for template in metadata.log_templates
        }

        # Create trace metadata with duration statistics
        metadata.traces = []
        for (span_name, service_name), duration_info in span_duration_stats.items():
            trace_meta = TraceMetadata(
                span_name=span_name,
                service_name=service_name,
                min_duration=duration_info["min"],
                max_duration=duration_info["max"],
            )
            metadata.traces.append(trace_meta)

        # Convert service calling edges from service names to service IDs
        metadata.service_calling_edges = []
        for edge in service_calling_edges:
            parent_service_name, child_service_name = edge
            if (
                parent_service_name in metadata.service_name_to_id
                and child_service_name in metadata.service_name_to_id
            ):
                parent_id = metadata.service_name_to_id[parent_service_name]
                child_id = metadata.service_name_to_id[child_service_name]
                metadata.service_calling_edges.append([parent_id, child_id])

        metadata.to_pkl(
            str(
                Path(self.config.get("paths.metadata")) / f"{self.dataset}_metadata.pkl"
            )
        )
        metadata.to_json(
            str(
                Path(self.config.get("paths.metadata"))
                / f"{self.dataset}_metadata.json"
            )
        )
        return metadata

    def create_metadata_for_folder(self, dataset_folder: str, label: str) -> DatasetMetadata:
        if label == "train":
            metadata = self.create_metadata()
            
            Path(".cache").mkdir(exist_ok=True)
            metadata.to_pkl(f".cache/{self.dataset}_{dataset_folder}_train_metadata.pkl")
            metadata.to_json(f".cache/{self.dataset}_{dataset_folder}_train_metadata.json")
            return metadata
        else:
            return DatasetMetadata(dataset_name=self.dataset)
