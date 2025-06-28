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
import json
from datetime import datetime, timedelta
import numpy as np
import pickle
from .utils import timeit
import gc
from functools import lru_cache
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading
import pytz
from rcabench.openapi import InjectionApi, ApiClient, Configuration
import random


class Processor(DataProcessor):
    def __init__(self, parsers: dict[str, BaseParser], conf: str = "config.toml"):
        config = self._load_config(conf)
        self.dataset: str = config.dataset  # type: ignore
        assert self.dataset != "", "Dataset name cannot be empty."
        self.config = config.datasets[config.dataset]  # type: ignore

        self.parsers: dict[str, BaseParser] = parsers
        self.drain = DrainProcessor(conf="drain.ini", save_path="cache/drain/temp")

        self._file_cache = {}
        self._lock = threading.Lock()  # Add thread safety for shared resources

        self.datapack_dir = self.config.root_path  # type: ignore
        assert isinstance(self.datapack_dir, str)
        assert self.datapack_dir != ""
        self.datapack_path = Path(self.datapack_dir)
        assert self.datapack_path.exists()
        assert self.datapack_path.is_dir()

        pl.Config.set_streaming_chunk_size(50000)
        pl.Config.set_tbl_rows(20)

    @lru_cache(maxsize=128)
    def _get_file_schema(self, file_path: str) -> dict:
        return pl.scan_parquet(file_path).collect_schema().to_python()

    def _optimize_memory_usage(self):
        gc.collect()
        if len(self._file_cache) > 100:
            self._file_cache.clear()

    def _load_config(self, config_path: str) -> Dynaconf:
        assert config_path != "", "Config path cannot be empty."
        return Dynaconf(settings_files=[config_path])

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

    def _load_datapacks(self, dataset) -> list[Path]:
        packs = []
        if self.config.load_method == "database":  # type: ignore
            packs = self._load_datapacks_from_db(dataset)
        elif self.config.load_method == "all":  # type: ignore
            packs = [dp for dp in self.datapack_path.iterdir() if dp.is_dir()]
        else:
            raise ValueError(
                f"Unsupported load method: {self.config.load_method}. "  # type: ignore
                "Supported methods are 'database' and 'all'."
            )

        valid_packs = []
        for pack in packs:
            if self._validate_datapack(pack):
                valid_packs.append(pack)

        if not valid_packs:
            raise ValueError("No valid datapacks found after validation.")

        logger.info(
            f"Loaded {len(valid_packs)} valid datapacks out of {len(packs)} total."
        )
        return valid_packs

    def _load_datapacks_from_db(self, dataset: str) -> list[Path]:
        config = Configuration(host="http://10.10.10.220:32080")
        with ApiClient(configuration=config) as client:
            api = InjectionApi(api_client=client)
            resp = api.api_v1_injections_analysis_with_issues_get()

        assert resp.data is not None, "No cases found in the response"
        case_names = list(
            set([item.injection_name for item in resp.data if item.injection_name])
        )
        return sorted([self.datapack_path / name / "converted" for name in case_names])

    def process_dataset(self):
        """
        1. create metadata: create_metadata
        2. process each datapack: process_datapack (with parallel processing)
           2.1 process log
           2.2 process metrics
           2.3 process traces
           2.4 load ground truth (labels)
        3. split to chunks(each chunk is the data within a certain time range, will be used for training/testing)
        """
        self.derive_files()
        self.datapacks = self._load_datapacks(self.dataset)
        self.metadata = self.create_metadata()

        samples: list[DataSample] = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all datapack processing tasks
            future_to_datapack = {
                executor.submit(self.process_datapack, datapack): datapack
                for datapack in self.datapacks
            }

            # Collect results with progress bar
            for future in tqdm(future_to_datapack, desc="Processing datapacks"):
                try:
                    datapack_samples = future.result()
                    samples.extend(datapack_samples)
                except Exception as e:
                    datapack = future_to_datapack[future]
                    logger.error(f"Error processing datapack {datapack}: {e}")
                    raise

        with open(f".cache/{self.dataset}_samples.pkl", "wb") as f:
            pickle.dump(samples, f)
        return samples

    def derive_files(self):
        def load(f) -> list[str]:
            files = self.config[f]  # type: ignore
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
        metadata = DatasetMetadata.from_pkl(f".cache/{self.dataset}_metadata.pkl")
        if metadata is None:
            metadata = DatasetMetadata(
                dataset_name=self.dataset,
            )

        all_metrics = set()
        all_services = set()
        all_span_names = set()
        all_log_templates = set()
        service_calling_edges = []
        metric_stats = {}

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
        metadata.traces = [
            TraceMetadata(span_name=span_name) for span_name in sorted(all_span_names)
        ]

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

        metadata.to_pkl(f".cache/{self.dataset}_metadata.pkl")
        metadata.to_json(f".cache/{self.dataset}_metadata.json")
        return metadata

    def extract_rcabench_labels(
        self, datapack: Path
    ) -> tuple[datetime, datetime, datetime, datetime, str, str]:
        label_file = self.config.label_files  # type: ignore

        assert label_file is not None, (
            f"Label file for {self.dataset} is not defined in the config."
        )
        assert isinstance(label_file, list)
        assert "injection.json" in label_file and "env.json" in label_file

        with open(datapack / "injection.json", "r") as f:
            injection = json.load(f)

        assert (
            injection is not None
            and "ground_truth" in injection
            and "service" in injection["ground_truth"]
        ), f"Invalid injection file format in {datapack / 'injection.json'}."
        gt_services = injection["ground_truth"]["service"]

        assert "fault_type" in injection
        fault_type = injection.get("fault_type")

        with open(datapack / "env.json", "r") as f:
            env = json.load(f)

        assert (
            "TIMEZONE" in env
            and "NORMAL_START" in env
            and "NORMAL_END" in env
            and "ABNORMAL_START" in env
            and "ABNORMAL_END" in env
        ), f"Invalid env file format in {datapack / 'env.json'}."

        tz = pytz.timezone(env["TIMEZONE"])
        normal_st = datetime.fromtimestamp(int(env["NORMAL_START"]), tz).astimezone(
            pytz.UTC
        )
        normal_et = datetime.fromtimestamp(int(env["NORMAL_END"]), tz).astimezone(
            pytz.UTC
        )
        abnormal_st = datetime.fromtimestamp(int(env["ABNORMAL_START"]), tz).astimezone(
            pytz.UTC
        )
        abnormal_et = datetime.fromtimestamp(int(env["ABNORMAL_END"]), tz).astimezone(
            pytz.UTC
        )

        return (
            normal_st,
            normal_et,
            abnormal_st,
            abnormal_et,
            gt_services[0],
            fault_type,
        )

    def extract_eadro_labels(
        self, datapack: Path
    ) -> tuple[datetime, datetime, datetime, datetime, str, str]:
        label_file = self.config.label_files  # type: ignore

        assert label_file is not None, (
            f"Label file for {self.dataset} is not defined in the config."
        )
        assert isinstance(label_file, list), "Label file should be a string."
        assert len(label_file) > 0, "No label file defined for datapack in the config."

        label_file_path = datapack / label_file[0]

        with open(label_file_path, "r") as f:
            label = json.load(f)

        assert isinstance(label, dict), "Label file should contain a dictionary."

        gt_service = label.get("injection_name", "")
        fault_type = label.get("fault_type", "")

        normal_st = (
            datetime.fromisoformat(label["normal_start_time"])
            if label.get("normal_start_time")
            else datetime(1970, 1, 1, 0, 0, 0)
        )
        normal_et = (
            datetime.fromisoformat(label["normal_end_time"])
            if label.get("normal_end_time")
            else datetime(1970, 1, 1, 0, 0, 0)
        )
        abnormal_st = (
            datetime.fromisoformat(label["fault_start_time"])
            if label.get("fault_start_time")
            else datetime(1970, 1, 1, 0, 0, 0)
        )
        abnormal_et = (
            datetime.fromisoformat(label["fault_end_time"])
            if label.get("fault_end_time")
            else datetime(1970, 1, 1)
        )

        return normal_st, normal_et, abnormal_st, abnormal_et, gt_service, fault_type

    def process_datapack(self, datapack: Path) -> list[DataSample]:
        samples = []

        if self.dataset == "rcabench":
            (
                normal_st,
                normal_et,
                abnormal_st,
                abnormal_et,
                gt_service,
                fault_type,
            ) = self.extract_rcabench_labels(datapack)
        elif self.dataset == "sn" or self.dataset == "tt":
            normal_st, normal_et, abnormal_st, abnormal_et, gt_service, fault_type = (
                self.extract_eadro_labels(datapack)
            )

        interval = self.config.sample_interval  # type: ignore
        sample_step = self.config.sample_step  # type: ignore
        assert isinstance(sample_step, int), "Sample step should be an integer."
        assert interval is not None, (
            f"Sample interval for {self.dataset} is not defined in the config."
        )
        assert isinstance(interval, int), "Sample interval should be an integer."

        log_dfs = {}
        metric_dfs = {}
        trace_dfs = {}

        time_range = []
        if normal_st != datetime(1970, 1, 1, 0, 0, 0) and normal_et != datetime(
            1970, 1, 1, 0, 0, 0
        ):
            time_range.append((normal_st, normal_et, "", ""))
        if abnormal_st != datetime(1970, 1, 1, 0, 0, 0) and abnormal_et != datetime(
            1970, 1, 1, 0, 0, 0
        ):
            time_range.append((abnormal_st, abnormal_et, fault_type, gt_service))

        for start, end, fault_type, gt_service in time_range:
            for log_file in self.log_files:
                log_dfs[log_file] = pl.scan_parquet(datapack / log_file).filter(
                    (pl.col("time") >= start) & (pl.col("time") <= end)
                )

            for metric_file in self.metric_files:
                metric_dfs[metric_file] = pl.scan_parquet(
                    datapack / metric_file
                ).filter((pl.col("time") >= start) & (pl.col("time") <= end))

            for trace_file in self.trace_files:
                trace_dfs[trace_file] = pl.scan_parquet(datapack / trace_file).filter(
                    (pl.col("time") >= start) & (pl.col("time") <= end)
                )

            current_time = start
            while current_time < end:
                window_end_time = current_time + timedelta(seconds=interval)
                if window_end_time > end:
                    break

                sample = DataSample(
                    abnormal=gt_service != "",
                    gt_service=gt_service,
                    fault_type=fault_type,
                )

                # Process log data
                for log_file in self.log_files:
                    window_df = log_dfs[log_file].filter(
                        (pl.col("time") >= current_time)
                        & (pl.col("time") <= window_end_time)
                    )
                    sample.log = self._process_log_from_df(
                        window_df, current_time, window_end_time
                    )
                    assert sample.log.shape == (
                        len(self.metadata.services),
                        len(self.metadata.log_templates) + 1,
                    )

                # Process metrics data
                for metric_file in self.metric_files:
                    window_df = metric_dfs[metric_file].filter(
                        (pl.col("time") >= current_time)
                        & (pl.col("time") <= window_end_time)
                    )
                    sample.metric = self._process_metrics_from_df(
                        window_df, current_time, window_end_time
                    )
                    assert sample.metric.shape == (
                        len(self.metadata.services),
                        interval,
                        len(self.metadata.metric_names),
                    )

                # Process trace data
                for trace_file in self.trace_files:
                    window_df = trace_dfs[trace_file].filter(
                        (pl.col("time") >= current_time)
                        & (pl.col("time") <= window_end_time)
                    )
                    sample.trace = self._process_traces_from_df(
                        window_df, current_time, window_end_time
                    )
                    assert sample.trace.shape == (
                        len(self.metadata.services),
                        interval,
                        2,
                    )

                samples.append(sample)
                current_time += timedelta(seconds=sample_step)

        return samples

    def _smooth_sparse_data(
        self, data: np.ndarray, num_services: int, interval: int, num_features: int
    ):
        """
        Smooth sparse time series data using forward fill and linear interpolation.

        Args:
            data: 3D array of shape (num_services, interval, num_features)
            num_services: Number of services
            interval: Number of time steps
            num_features: Number of features
        """
        for service_id in range(num_services):
            for feature_id in range(num_features):
                # Get the time series for this service-feature combination
                time_series = data[service_id, :, feature_id]

                # Find non-zero indices (assuming zero means no data)
                non_zero_indices = np.where(time_series != 0)[0]

                if len(non_zero_indices) == 0:
                    # No data points, keep as zeros
                    continue
                elif len(non_zero_indices) == 1:
                    # Only one data point, forward fill
                    single_value = time_series[non_zero_indices[0]]
                    data[service_id, non_zero_indices[0] :, feature_id] = single_value
                else:
                    # Multiple data points, use linear interpolation
                    for i in range(len(non_zero_indices) - 1):
                        start_idx = non_zero_indices[i]
                        end_idx = non_zero_indices[i + 1]
                        start_val = time_series[start_idx]
                        end_val = time_series[end_idx]

                        # Linear interpolation between data points
                        if end_idx - start_idx > 1:
                            interpolated_values = np.linspace(
                                start_val, end_val, end_idx - start_idx + 1
                            )
                            data[service_id, start_idx : end_idx + 1, feature_id] = (
                                interpolated_values
                            )

                    # Forward fill from the last data point to the end
                    last_idx = non_zero_indices[-1]
                    last_val = time_series[last_idx]
                    data[service_id, last_idx:, feature_id] = last_val

                    # Backward fill from the first data point to the beginning
                    first_idx = non_zero_indices[0]
                    first_val = time_series[first_idx]
                    data[service_id, : first_idx + 1, feature_id] = first_val

    def _process_log_from_df(
        self, df: pl.LazyFrame, start_time: datetime, end_time: datetime
    ) -> np.ndarray:
        """Helper method to process log data from a pre-filtered DataFrame"""
        num_services = len(self.metadata.services)
        num_templates = len(self.metadata.log_templates) + 1  # +1 for unseen templates

        # Collect data and check if empty
        df_collected = df.select(["message", "service_name"]).collect()
        if df_collected.height == 0:
            result = np.zeros((num_services, num_templates))
            assert result.shape == (num_services, num_templates), (
                f"Expected log result shape ({num_services}, {num_templates}), "
                f"got {result.shape}"
            )
            return result

        # Batch process messages
        messages = df_collected["message"].to_list()
        templates = self.drain.process_batch(messages)

        # Add template column
        df_with_templates = df_collected.with_columns(pl.Series("template", templates))

        result = np.zeros((num_services, num_templates))

        # Calculate counts
        counts = df_with_templates.group_by(["service_name", "template"]).agg(
            pl.count().alias("count")
        )

        service_lookup = self.metadata.service_name_to_id
        template_lookup = self.metadata.log_template_to_id

        for row in counts.iter_rows(named=True):
            service_name = row["service_name"]
            template = row["template"]
            count = row["count"]

            service_id = service_lookup.get(service_name)
            if service_id is not None:
                template_id = template_lookup.get(template, 0)
                assert 0 <= template_id < num_templates, (
                    f"Template ID {template_id} out of bounds [0, {num_templates})"
                )
                result[service_id, template_id] += count

        assert result.shape == (num_services, num_templates), (
            f"Expected log result shape ({num_services}, {num_templates}), "
            f"got {result.shape}"
        )
        return result

    def _process_metrics_from_df(
        self, df: pl.LazyFrame, start_time: datetime, end_time: datetime
    ) -> np.ndarray:
        """Helper method to process metrics data from a pre-filtered DataFrame"""
        interval = int(self.config.sample_interval)  # type: ignore
        num_services = len(self.metadata.services)
        num_metrics = len(self.metadata.metric_names)

        # Calculate time buckets
        window_duration = (end_time - start_time).total_seconds()
        time_step_size = window_duration / interval if interval > 0 else 1.0

        # Add time buckets and collect data
        df_with_buckets = (
            df.with_columns(
                [
                    ((pl.col("time") - start_time).dt.total_seconds() / time_step_size)
                    .floor()
                    .cast(pl.Int32)
                    .alias("time_bucket")
                ]
            )
            .filter((pl.col("time_bucket") >= 0) & (pl.col("time_bucket") < interval))
            .collect()
        )

        if df_with_buckets.height == 0:
            result = np.zeros((num_services, interval, num_metrics))
            assert result.shape == (num_services, interval, num_metrics), (
                f"Expected metrics result shape ({num_services}, {interval}, {num_metrics}), "
                f"got {result.shape}"
            )
            return result

        result = np.zeros((num_services, interval, num_metrics))

        # Calculate statistics
        metrics_stats = df_with_buckets.group_by(
            ["service_name", "metric", "time_bucket"]
        ).agg(pl.col("value").mean().alias("mean_value"))

        service_lookup = self.metadata.service_name_to_id
        metric_lookup = self.metadata.metric_name_to_id

        for row in metrics_stats.iter_rows(named=True):
            service_name = row["service_name"]
            metric_name = row["metric"]
            time_bucket = row["time_bucket"]
            mean_value = row["mean_value"]

            service_id = service_lookup.get(service_name)
            metric_id = metric_lookup.get(metric_name)

            if (
                service_id is not None
                and metric_id is not None
                and 0 <= time_bucket < interval
            ):
                # Normalization processing
                metric_meta = self.metadata.metrics[metric_id]
                if (
                    metric_meta.min_value is not None
                    and metric_meta.max_value is not None
                ):
                    value_range = metric_meta.max_value - metric_meta.min_value
                    if value_range > 0:
                        normalized_value = (
                            mean_value - metric_meta.min_value
                        ) / value_range
                    else:
                        normalized_value = 0.0
                else:
                    normalized_value = mean_value

                result[service_id, time_bucket, metric_id] = normalized_value

        # Data smoothing
        self._smooth_sparse_data(result, num_services, interval, num_metrics)

        assert result.shape == (num_services, interval, num_metrics), (
            f"Expected metrics result shape ({num_services}, {interval}, {num_metrics}), "
            f"got {result.shape}"
        )
        return result

    def _process_traces_from_df(
        self, df: pl.LazyFrame, start_time: datetime, end_time: datetime
    ) -> np.ndarray:
        """Helper method to process traces data from a pre-filtered DataFrame"""
        interval = int(self.config.sample_interval)  # type: ignore
        num_services = len(self.metadata.services)
        expected_features = 2  # latency and invocation count

        # Calculate time buckets
        window_duration = (end_time - start_time).total_seconds()
        time_step_size = window_duration / interval if interval > 0 else 1.0

        # Add time buckets and collect data
        df_with_buckets = (
            df.with_columns(
                [
                    ((pl.col("time") - start_time).dt.total_seconds() / time_step_size)
                    .floor()
                    .cast(pl.Int32)
                    .alias("time_bucket")
                ]
            )
            .filter((pl.col("time_bucket") >= 0) & (pl.col("time_bucket") < interval))
            .collect()
        )

        if df_with_buckets.height == 0:
            result = np.zeros((num_services, interval, expected_features))
            assert result.shape == (num_services, interval, expected_features), (
                f"Expected traces result shape ({num_services}, {interval}, {expected_features}), "
                f"got {result.shape}"
            )
            return result

        result = np.zeros((num_services, interval, expected_features))

        # Calculate latency and invocation statistics
        trace_stats = df_with_buckets.group_by(["service_name", "time_bucket"]).agg(
            [
                pl.col("duration").mean().alias("avg_latency"),
                pl.col("duration").count().alias("invocation_count"),
            ]
        )

        service_lookup = self.metadata.service_name_to_id

        for row in trace_stats.iter_rows(named=True):
            service_name = row["service_name"]
            time_bucket = row["time_bucket"]
            avg_latency = row["avg_latency"]
            invocation_count = row["invocation_count"]

            service_id = service_lookup.get(service_name)
            if service_id is not None and 0 <= time_bucket < interval:
                result[service_id, time_bucket, 0] = (
                    avg_latency if avg_latency is not None else 0.0
                )
                result[service_id, time_bucket, 1] = (
                    invocation_count if invocation_count is not None else 0.0
                )

        # Data smoothing
        self._smooth_sparse_data(result, num_services, interval, expected_features)

        # Z-score normalization
        latency_values = result[:, :, 0].flatten()
        if np.std(latency_values) > 1e-8:
            mean_latency = np.mean(latency_values)
            std_latency = np.std(latency_values)
            result[:, :, 0] = (result[:, :, 0] - mean_latency) / std_latency

        assert result.shape == (num_services, interval, expected_features), (
            f"Expected traces result shape ({num_services}, {interval}, {expected_features}), "
            f"got {result.shape}"
        )
        return result
