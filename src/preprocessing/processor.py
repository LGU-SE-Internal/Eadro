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
from .parsers.log import DrainProcessor
import json
from datetime import datetime, timedelta
import numpy as np
import pickle


class Processor(DataProcessor):
    def __init__(self, parsers: dict[str, BaseParser], conf: str = "config.toml"):
        config = self._load_config(conf)
        self.dataset: str = config.dataset  # type: ignore
        assert self.dataset != "", "Dataset name cannot be empty."
        self.config = config.datasets[config.dataset]  # type: ignore

        self.parsers: dict[str, BaseParser] = parsers
        self.drain = DrainProcessor(conf="drain.ini", save_path="cache/drain/temp")

    def _load_config(self, config_path: str) -> Dynaconf:
        assert config_path != "", "Config path cannot be empty."
        return Dynaconf(settings_files=[config_path])

    def _load_datapacks(self, dataset) -> list[Path]:
        datapack_dir = self.config.root_path  # type: ignore
        assert isinstance(datapack_dir, str), (
            f"Datapack directory for {dataset} should be a string, got {type(datapack_dir)}."
        )
        assert datapack_dir != "", (
            f"Datapack directory for {dataset} is not set in the config."
        )
        datapack_path = Path(datapack_dir)
        assert datapack_path.exists(), (
            f"Datapack directory {datapack_path} does not exist."
        )
        assert datapack_path.is_dir(), (
            f"Datapack directory {datapack_path} is not a directory."
        )
        return [dp for dp in datapack_path.iterdir() if dp.is_dir()]

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
        self.datapacks = self._load_datapacks(self.dataset)
        self.derive_files()
        self.metadata = self.create_metadata()

        samples: list[DataSample] = []
        for datapack in tqdm(self.datapacks, desc="Processing datapacks"):
            samples.extend(self.process_datapack(datapack))

        # dict_samples = {np.int64(i): sample for i, sample in enumerate(samples)}
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

        for datapack in tqdm(self.datapacks, desc="Processing datapacks"):
            for metric_file in self.metric_files:
                df = pl.scan_parquet(datapack / metric_file)

                unique_metrics = df.select(pl.col("metric")).unique().collect()
                all_metrics.update(unique_metrics["metric"].to_list())
                unique_services = df.select(pl.col("service_name")).unique().collect()
                all_services.update(unique_services["service_name"].to_list())

                stats = (
                    df.group_by("metric")
                    .agg(
                        [
                            pl.col("value").min().alias("min_value"),
                            pl.col("value").max().alias("max_value"),
                        ]
                    )
                    .collect()
                )

                for row in stats.iter_rows(named=True):
                    metric_name = row["metric"]
                    min_val = row["min_value"]
                    max_val = row["max_value"]

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
                df = pl.scan_parquet(datapack / log_file)
                unique_services = df.select(pl.col("service_name")).unique().collect()
                all_services.update(unique_services["service_name"].to_list())

                logs = (
                    df.select(pl.col("message")).unique().collect()["message"].to_list()
                )
                all_log_templates.update(self.drain.process_batch(logs))

            for trace_file in self.trace_files:
                df = pl.scan_parquet(datapack / trace_file)
                unique_services = df.select(pl.col("service_name")).unique().collect()
                all_services.update(unique_services["service_name"].to_list())

                unique_span_names = df.select(pl.col("span_name")).unique().collect()
                all_span_names.update(unique_span_names["span_name"].to_list())

            for trace_file in self.trace_files:
                df = pl.scan_parquet(datapack / trace_file)
                assert (
                    "span_id" in df.collect_schema().names()
                    and "parent_span_id" in df.collect_schema().names()
                ), (
                    f"Trace data in {trace_file} does not contain 'span_id' or 'parent_span_id' columns."
                )

                spans_df = df.select(
                    [
                        pl.col("span_id"),
                        pl.col("parent_span_id"),
                        pl.col("service_name"),
                    ]
                ).collect()

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

                for row in calling_edges.iter_rows(named=True):
                    parent_service = row["parent_service"]
                    child_service = row["child_service"]
                    if parent_service != child_service:
                        edge = [parent_service, child_service]
                        if edge not in service_calling_edges:
                            service_calling_edges.append(edge)

            # for label_file in self.config.label_files:  # type: ignore
            #     fault_type = self.parsers[f"{self.dataset}_fault"].parse(
            #         datapack / label_file
            #     )
            #     all_fault_types.add(fault_type)

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

    def extract_labels(self, datapack: Path) -> tuple[datetime, datetime, str, str]:
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
        fault_type = ""
        if gt_service == "":
            start_time = label.get("normal_start_time", "")
            end_time = label.get("normal_end_time", "")
        else:
            start_time = label.get("fault_start_time", "")
            end_time = label.get("fault_end_time", "")
            fault_type = label.get("fault_type", "")

        st = datetime.fromisoformat(start_time)
        et = datetime.fromisoformat(end_time)
        return st, et, gt_service, fault_type

    def process_datapack(self, datapack: Path) -> list[DataSample]:
        samples = []
        start_time, end_time, gt_service, fault_type = self.extract_labels(datapack)

        interval = self.config.sample_interval  # type: ignore
        assert interval is not None, (
            f"Sample interval for {self.dataset} is not defined in the config."
        )
        assert isinstance(interval, int), "Sample interval should be an integer."

        current_time = start_time
        while current_time < end_time:
            window_end_time = current_time + timedelta(seconds=interval)
            if window_end_time > end_time:
                break

            sample = DataSample(
                abnormal=fault_type != "", gt_service=gt_service, fault_type=fault_type
            )
            for log in self.log_files:
                sample.log = self.process_log(
                    datapack / log, current_time, window_end_time
                )
                assert sample.log.shape == (
                    len(self.metadata.services),
                    len(self.metadata.log_templates) + 1,
                )

            for metric in self.metric_files:
                sample.metric = self.process_metrics(
                    datapack / metric, current_time, window_end_time
                )
                assert sample.metric.shape == (
                    len(self.metadata.services),
                    len(self.metadata.metric_names),
                )

            for trace in self.trace_files:
                sample.trace = self.process_traces(
                    datapack / trace, current_time, window_end_time
                )
                assert sample.trace.shape == (
                    len(self.metadata.services),
                    2,
                )

            samples.append(sample)

            current_time += timedelta(seconds=interval)

        return samples

    def process_log(
        self,
        log_path: Path,
        start_time: datetime,
        end_time: datetime,
    ) -> np.ndarray:
        """
        Process log data within a time window.
        Returns: np.ndarray of shape (num_services, num_templates)
        """
        df = pl.read_parquet(log_path)

        df = df.filter((pl.col("time") >= start_time) & (pl.col("time") <= end_time))

        num_services = len(self.metadata.services)
        num_templates = len(self.metadata.log_templates) + 1  # +1 for unseen templates

        if df.height == 0:
            result = np.zeros((num_services, num_templates))
            # Assert dimensions for empty result
            assert result.shape == (num_services, num_templates), (
                f"Expected log result shape ({num_services}, {num_templates}), "
                f"got {result.shape}"
            )
            return result

        messages = df.select("message").to_pandas()["message"].tolist()
        templates = self.drain.process_batch(messages)

        # Add templates to dataframe
        df = df.with_columns(pl.Series("template", templates))

        result = np.zeros((num_services, num_templates))

        assert "service_name" in df.columns, (
            f"Log data in {log_path} does not contain 'service_name' column."
        )
        counts = df.group_by(["service_name", "template"]).agg(
            pl.count().alias("count")
        )

        for row in counts.iter_rows(named=True):
            service_name = row["service_name"]
            template = row["template"]
            count = row["count"]

            if service_name in self.metadata.service_name_to_id:
                service_id = self.metadata.service_name_to_id[service_name]
                # Assert service_id is within bounds
                assert 0 <= service_id < num_services, (
                    f"Service ID {service_id} out of bounds [0, {num_services})"
                )

                template_id = self.metadata.log_template_to_id.get(template, 0)
                # Assert template_id is within bounds
                assert 0 <= template_id < num_templates, (
                    f"Template ID {template_id} out of bounds [0, {num_templates})"
                )

                result[service_id, template_id] += count

        # Final dimension check
        assert result.shape == (num_services, num_templates), (
            f"Expected log result shape ({num_services}, {num_templates}), "
            f"got {result.shape}"
        )

        return result

    def process_metrics(
        self,
        metric_path: Path,
        start_time: datetime,
        end_time: datetime,
    ) -> np.ndarray:
        """
        Process metrics data within a time window.
        Returns: np.ndarray of shape (num_services, num_metrics)
        """
        df = pl.read_parquet(metric_path)

        # Filter by time range
        df = df.filter((pl.col("time") >= start_time) & (pl.col("time") <= end_time))

        # Initialize result array: (num_services, num_metrics)
        num_services = len(self.metadata.services)
        num_metrics = len(self.metadata.metric_names)

        if df.height == 0:
            # No metrics in this time window
            result = np.zeros((num_services, num_metrics))
            # Assert dimensions for empty result
            assert result.shape == (num_services, num_metrics), (
                f"Expected metrics result shape ({num_services}, {num_metrics}), "
                f"got {result.shape}"
            )
            return result

        result = np.zeros((num_services, num_metrics))

        # Group by service and metric, calculate mean values
        assert "service_name" in df.columns, (
            f"Metrics data in {metric_path} does not contain 'service_name' column."
        )
        assert "metric" in df.columns, (
            f"Metrics data in {metric_path} does not contain 'metric' column."
        )
        assert "value" in df.columns, (
            f"Metrics data in {metric_path} does not contain 'value' column."
        )

        # Calculate mean values for each service-metric combination
        metrics_stats = df.group_by(["service_name", "metric"]).agg(
            pl.col("value").mean().alias("mean_value")
        )

        for row in metrics_stats.iter_rows(named=True):
            service_name = row["service_name"]
            metric_name = row["metric"]
            mean_value = row["mean_value"]

            if (
                service_name in self.metadata.service_name_to_id
                and metric_name in self.metadata.metric_name_to_id
            ):
                service_id = self.metadata.service_name_to_id[service_name]
                metric_id = self.metadata.metric_name_to_id[metric_name]

                # Assert indices are within bounds
                assert 0 <= service_id < num_services, (
                    f"Service ID {service_id} out of bounds [0, {num_services})"
                )
                assert 0 <= metric_id < num_metrics, (
                    f"Metric ID {metric_id} out of bounds [0, {num_metrics})"
                )

                # Normalize value using min-max normalization
                metric_meta = self.metadata.metrics[metric_id]
                if (
                    metric_meta.min_value is not None
                    and metric_meta.max_value is not None
                ):
                    # Simple min-max normalization
                    value_range = metric_meta.max_value - metric_meta.min_value
                    if value_range > 0:
                        normalized_value = (
                            mean_value - metric_meta.min_value
                        ) / value_range
                    else:
                        normalized_value = 0.0
                else:
                    normalized_value = mean_value

                result[service_id, metric_id] = normalized_value

        # Final dimension check
        assert result.shape == (num_services, num_metrics), (
            f"Expected metrics result shape ({num_services}, {num_metrics}), "
            f"got {result.shape}"
        )

        return result

    def process_traces(
        self,
        trace_path: Path,
        start_time: datetime,
        end_time: datetime,
    ) -> np.ndarray:
        """
        Process trace data within a time window.
        Returns: np.ndarray of shape (num_services, 2) where the 2 features are:
        - 0: average latency for the service
        - 1: number of invocations for the service
        """
        df = pl.read_parquet(trace_path)

        # Filter by time range
        df = df.filter((pl.col("time") >= start_time) & (pl.col("time") <= end_time))

        # Initialize result array: (num_services, 2)
        # Feature 0: average latency, Feature 1: number of invocations
        num_services = len(self.metadata.services)
        expected_features = 2  # latency and invocation count

        if df.height == 0:
            # No traces in this time window
            result = np.zeros((num_services, expected_features))
            # Assert dimensions for empty result
            assert result.shape == (num_services, expected_features), (
                f"Expected traces result shape ({num_services}, {expected_features}), "
                f"got {result.shape}"
            )
            return result

        result = np.zeros((num_services, expected_features))

        assert "service_name" in df.columns, (
            f"Trace data in {trace_path} does not contain 'service_name' column."
        )
        assert "duration" in df.columns, (
            f"Trace data in {trace_path} does not contain 'duration' column."
        )

        # Calculate latency and invocation statistics per service
        trace_stats = df.group_by(["service_name"]).agg(
            [
                pl.col("duration").mean().alias("avg_latency"),
                pl.col("duration").count().alias("invocation_count"),
            ]
        )

        for row in trace_stats.iter_rows(named=True):
            service_name = row["service_name"]
            avg_latency = row["avg_latency"]
            invocation_count = row["invocation_count"]

            if service_name in self.metadata.service_name_to_id:
                service_id = self.metadata.service_name_to_id[service_name]

                # Assert service_id is within bounds
                assert 0 <= service_id < num_services, (
                    f"Service ID {service_id} out of bounds [0, {num_services})"
                )

                result[service_id, 0] = avg_latency if avg_latency is not None else 0.0
                result[service_id, 1] = (
                    invocation_count if invocation_count is not None else 0.0
                )

        # Z-score normalization for latency values
        latency_values = result[:, 0]
        if np.std(latency_values) > 1e-8:
            result[:, 0] = (latency_values - np.mean(latency_values)) / np.std(
                latency_values
            )

        # Final dimension check
        assert result.shape == (num_services, expected_features), (
            f"Expected traces result shape ({num_services}, {expected_features}), "
            f"got {result.shape}"
        )

        return result
