from .base import (
    DataProcessor,
    DatasetMetadata,
    TimeSeriesDataSample,
    TimeSeriesDatapack,
    BaseParser,
    ServiceMetadata,
    MetricMetadata,
    LogTemplateMetadata,
    TraceMetadata,
)
from dynaconf import Dynaconf
from pathlib import Path
import polars as pl
from tqdm import tqdm
from .log import DrainProcessor
import json
from datetime import datetime
import numpy as np
import gc
from functools import lru_cache
from loguru import logger
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
        config = Configuration(host="http://10.10.10.161:8082")
        with ApiClient(configuration=config) as client:
            api = InjectionApi(api_client=client)
            resp = api.api_v1_injections_analysis_with_issues_get()

        assert resp.data is not None, "No cases found in the response"
        case_names = list(
            set([item.injection_name for item in resp.data if item.injection_name])
        )
        return sorted([self.datapack_path / name / "converted" for name in case_names])

    def process_dataset(self) -> TimeSeriesDatapack:
        self.derive_files()
        self.datapacks = self._load_datapacks(self.dataset)
        self.metadata = self.create_metadata()

        time_series_samples: list[TimeSeriesDataSample] = []

        for datapack in tqdm(self.datapacks, desc="Processing datapacks (continuous)"):
            ts_sample = self.process_datapack(datapack)
            if ts_sample.get_time_steps() > 0:
                time_series_samples.append(ts_sample)
            else:
                logger.warning(f"Skipping datapack {datapack} - no valid time steps")

        datapack = TimeSeriesDatapack(
            samples=time_series_samples, metadata=self.metadata
        )

        datapack_path = f".cache/{self.dataset}_timeseries_datapack.pkl"
        datapack.save(datapack_path)
        logger.info(
            f"Saved {len(time_series_samples)} time series samples to {datapack_path}"
        )

        return datapack

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
        else:
            logger.info(f"Loaded existing metadata for {self.dataset} from cache.")
            return metadata

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

    def process_datapack(self, datapack: Path) -> TimeSeriesDataSample:
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
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        start_time, end_time = self._get_overall_time_range(
            normal_st, normal_et, abnormal_st, abnormal_et
        )
        assert start_time < end_time, (
            f"Invalid time range for datapack {datapack}: {start_time} >= {end_time}"
        )

        time_resolution = 1
        time_steps = int((end_time - start_time).total_seconds() / time_resolution)

        log_series = self._process_logs(datapack, start_time, end_time, time_steps)
        metric_series = self._process_metrics(
            datapack, start_time, end_time, time_steps
        )
        trace_series = self._process_traces(datapack, start_time, end_time, time_steps)

        normal_periods = []
        abnormal_periods = []

        if normal_st != datetime(1970, 1, 1, 0, 0, 0) and normal_et != datetime(
            1970, 1, 1, 0, 0, 0
        ):
            normal_periods.append((normal_st, normal_et))

        if abnormal_st != datetime(1970, 1, 1, 0, 0, 0) and abnormal_et != datetime(
            1970, 1, 1, 0, 0, 0
        ):
            abnormal_periods.append((abnormal_st, abnormal_et, gt_service, fault_type))

        return TimeSeriesDataSample(
            datapack_name=datapack.name,
            start_time=start_time,
            end_time=end_time,
            time_resolution=time_resolution,
            normal_periods=normal_periods,
            abnormal_periods=abnormal_periods,
            log_series=log_series,
            metric_series=metric_series,
            trace_series=trace_series,
        )

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
            else datetime(1970, 1, 1, 0, 0, 0)
        )

        return normal_st, normal_et, abnormal_st, abnormal_et, gt_service, fault_type

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

    def _get_overall_time_range(
        self,
        normal_st: datetime,
        normal_et: datetime,
        abnormal_st: datetime,
        abnormal_et: datetime,
    ) -> tuple[datetime, datetime]:
        times = []

        if normal_st != datetime(1970, 1, 1, 0, 0, 0):
            times.append(normal_st)
        if normal_et != datetime(1970, 1, 1, 0, 0, 0):
            times.append(normal_et)
        if abnormal_st != datetime(1970, 1, 1, 0, 0, 0):
            times.append(abnormal_st)
        if abnormal_et != datetime(1970, 1, 1, 0, 0, 0):
            times.append(abnormal_et)

        if not times:
            return datetime(1970, 1, 1, 0, 0, 0), datetime(1970, 1, 1, 0, 0, 1)

        return min(times), max(times)

    def _process_logs(
        self, datapack: Path, start_time: datetime, end_time: datetime, time_steps: int
    ) -> np.ndarray:
        num_services = len(self.metadata.services)
        num_templates = len(self.metadata.log_templates) + 1

        result = np.zeros((time_steps, num_services, num_templates))

        for log_file in self.log_files:
            log_path = datapack / log_file
            assert log_path.exists()

            df = (
                pl.scan_parquet(log_path)
                .filter((pl.col("time") >= start_time) & (pl.col("time") <= end_time))
                .select(["time", "message", "service_name"])
                .collect()
            )

            if df.height == 0:
                continue

            messages = df["message"].to_list()
            templates = self.drain.process_batch(messages)

            df_with_templates = df.with_columns(pl.Series("template", templates))

            time_deltas = (df_with_templates["time"] - start_time).dt.total_seconds()
            time_indices = (time_deltas // 1).cast(pl.Int32)

            valid_mask = (time_indices >= 0) & (time_indices < time_steps)
            df_filtered = df_with_templates.filter(valid_mask)
            time_indices_filtered = time_indices.filter(valid_mask)

            assert df_filtered.height > 0

            df_with_time_idx = df_filtered.with_columns(
                pl.Series("time_idx", time_indices_filtered.to_list())
            )

            counts = df_with_time_idx.group_by(
                ["time_idx", "service_name", "template"]
            ).agg(pl.count().alias("count"))

            service_lookup = self.metadata.service_name_to_id
            template_lookup = self.metadata.log_template_to_id

            for row in counts.iter_rows(named=True):
                time_idx = row["time_idx"]
                service_name = row["service_name"]
                template = row["template"]
                count = row["count"]

                service_id = service_lookup.get(service_name)
                if service_id is not None:
                    template_id = template_lookup.get(
                        template, len(self.metadata.log_templates)
                    )
                    if 0 <= time_idx < time_steps and 0 <= template_id < num_templates:
                        result[time_idx, service_id, template_id] += count

        return result

    def _process_metrics(
        self, datapack: Path, start_time: datetime, end_time: datetime, time_steps: int
    ) -> np.ndarray:
        num_services = len(self.metadata.services)
        num_metrics = len(self.metadata.metric_names)

        # (time_steps, services, metrics)
        result = np.zeros((time_steps, num_services, num_metrics))

        for metric_file in self.metric_files:
            metric_path = datapack / metric_file
            if not metric_path.exists():
                logger.warning(f"Metric file {metric_path} does not exist")
                continue

            df = (
                pl.scan_parquet(metric_path)
                .filter((pl.col("time") >= start_time) & (pl.col("time") <= end_time))
                .select(["time", "metric", "service_name", "value"])
                .collect()
            )

            if df.height == 0:
                continue

            time_deltas = (df["time"] - start_time).dt.total_seconds()
            time_indices = (time_deltas // 1).cast(pl.Int32)  # 1秒分辨率

            valid_mask = (time_indices >= 0) & (time_indices < time_steps)
            df_filtered = df.filter(valid_mask)
            time_indices_filtered = time_indices.filter(valid_mask)

            if df_filtered.height == 0:
                continue

            df_with_time_idx = df_filtered.with_columns(
                pl.Series("time_idx", time_indices_filtered.to_list())
            )

            metrics_stats = df_with_time_idx.group_by(
                ["time_idx", "service_name", "metric"]
            ).agg(pl.col("value").mean().alias("mean_value"))

            service_lookup = self.metadata.service_name_to_id
            metric_lookup = self.metadata.metric_name_to_id

            for row in metrics_stats.iter_rows(named=True):
                time_idx = row["time_idx"]
                service_name = row["service_name"]
                metric_name = row["metric"]
                mean_value = row["mean_value"]

                service_id = service_lookup.get(service_name)
                metric_id = metric_lookup.get(metric_name)

                if (
                    service_id is not None
                    and metric_id is not None
                    and 0 <= time_idx < time_steps
                    and mean_value is not None
                ):
                    result[time_idx, service_id, metric_id] = mean_value

        self._smooth_sparse_data(result, num_services, time_steps, num_metrics)
        return result

    def _process_traces(
        self, datapack: Path, start_time: datetime, end_time: datetime, time_steps: int
    ) -> np.ndarray:
        num_services = len(self.metadata.services)
        trace_features = 2  #  span_count, avg_duration

        result = np.zeros((time_steps, num_services, trace_features))

        for trace_file in self.trace_files:
            trace_path = datapack / trace_file
            if not trace_path.exists():
                logger.warning(f"Trace file {trace_path} does not exist")
                continue

            df = (
                pl.scan_parquet(trace_path)
                .filter((pl.col("time") >= start_time) & (pl.col("time") <= end_time))
                .select(["time", "service_name", "duration"])
                .collect()
            )

            if df.height == 0:
                continue

            time_deltas = (df["time"] - start_time).dt.total_seconds()
            time_indices = (time_deltas // 1).cast(pl.Int32)  # 1秒分辨率

            valid_mask = (time_indices >= 0) & (time_indices < time_steps)
            df_filtered = df.filter(valid_mask)
            time_indices_filtered = time_indices.filter(valid_mask)

            if df_filtered.height == 0:
                continue

            df_with_time_idx = df_filtered.with_columns(
                pl.Series("time_idx", time_indices_filtered.to_list())
            )

            trace_stats = df_with_time_idx.group_by(["time_idx", "service_name"]).agg(
                [
                    pl.count().alias("span_count"),
                    pl.col("duration").mean().alias("avg_duration"),
                ]
            )

            service_lookup = self.metadata.service_name_to_id

            for row in trace_stats.iter_rows(named=True):
                time_idx = row["time_idx"]
                service_name = row["service_name"]
                span_count = row["span_count"]
                avg_duration = row["avg_duration"]

                service_id = service_lookup.get(service_name)
                if service_id is not None and 0 <= time_idx < time_steps:
                    result[time_idx, service_id, 0] = span_count
                    if avg_duration is not None:
                        result[time_idx, service_id, 1] = avg_duration

        self._smooth_sparse_data(result, num_services, time_steps, trace_features)
        return result
