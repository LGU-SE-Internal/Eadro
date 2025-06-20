import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from utils import CacheManager
import logging


class DrainProcesser:
    def __init__(self, conf: str, save_path: str, cache_dir: str = "./cache/drain"):
        persistence = FilePersistence(save_path)
        miner_config = TemplateMinerConfig()
        miner_config.load(conf)
        self._template_miner = TemplateMiner(persistence, config=miner_config)
        self._cache_manager = CacheManager[str](
            Path(cache_dir) / "sentence_templates.pkl"
        )

    def __call__(self, sentence: str) -> str:
        line = str(sentence).strip()
        if not line:
            return ""

        return self._cache_manager.get_or_compute(
            line, lambda: self._process_line(line)
        )

    def _process_line(self, line: str) -> str:
        result = self._template_miner.add_log_message(line)
        template = result.get("template_mined")
        if template is None:
            logging.warning(
                f"Failed to find 'template_mined' for line: {line}. Result: {result}"
            )
            return ""
        return template

    def save_cache(self):
        self._cache_manager.save()


class DataAdapter:
    def __init__(self, chunk_length: int = 1, threshold: int = 1):
        self.chunk_length = chunk_length
        self.threshold = threshold
        self.service2node_id = {}
        self.node_id2service = {}
        self.log_templates = []
        self.metric_names = []

        self.drain = DrainProcesser(
            conf="drain.ini",
            save_path="data/drain_templates",
            cache_dir="data/cache/drain",
        )

    def derive_filename(self, data_pack: Path) -> Dict[str, Path]:
        return {
            "abnormal_log": data_pack / "abnormal_logs.parquet",
            "normal_log": data_pack / "normal_logs.parquet",
            "abnormal_metric": data_pack / "abnormal_metrics.parquet",
            "normal_metric": data_pack / "normal_metrics.parquet",
            "abnormal_trace": data_pack / "abnormal_traces.parquet",
            "normal_trace": data_pack / "normal_traces.parquet",
            "env": data_pack / "env.json",
            "injection": data_pack / "injection.json",
        }

    def build_service_mapping(self, data_files: Dict[str, Path]) -> None:
        all_services = set()

        for file_type in [
            "normal_log",
            "abnormal_log",
            "normal_metric",
            "abnormal_metric",
            "normal_trace",
            "abnormal_trace",
        ]:
            if data_files[file_type].exists():
                df = pd.read_parquet(data_files[file_type])
                if "service_name" in df.columns:
                    all_services.update(
                        df[df["service_name"].notna()]["service_name"].unique()
                    )

        self.service2node_id = {
            service: idx for idx, service in enumerate(sorted(all_services))
        }
        self.node_id2service = {
            idx: service for service, idx in self.service2node_id.items()
        }

        print(f"Found {len(all_services)} unique services")

    def extract_log_templates(self, data_files: Dict[str, Path]) -> None:
        all_messages = []

        for file_type in ["normal_log", "abnormal_log"]:
            if data_files[file_type].exists():
                df = pd.read_parquet(data_files[file_type])
                if "message" in df.columns:
                    all_messages.extend(
                        [self.drain(i) for i in df["message"].astype(str).tolist()]
                    )

        message_counts = Counter(all_messages)
        self.log_templates = [msg for msg, count in message_counts.most_common(100)]

        print(f"Extracted {len(self.log_templates)} log templates")

    def extract_metric_names(self, data_files: Dict[str, Path]) -> None:
        all_metrics = set()

        for file_type in ["normal_metric", "abnormal_metric"]:
            if data_files[file_type].exists():
                df = pd.read_parquet(data_files[file_type])
                if "metric" in df.columns:
                    all_metrics.update(df["metric"].unique())

        self.metric_names = sorted(list(all_metrics))
        print(f"Found {len(self.metric_names)} unique metrics")

    def get_time_intervals(
        self, data_files: Dict[str, Path]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        with open(data_files["env"]) as f:
            env_data = json.load(f)

        start_time = pd.to_datetime(int(env_data["NORMAL_START"]), unit="s", utc=True)
        end_time = pd.to_datetime(int(env_data["ABNORMAL_END"]), unit="s", utc=True)

        intervals = []
        current_time = start_time

        while current_time < end_time:
            window_end = current_time + pd.Timedelta(minutes=self.chunk_length)
            intervals.append((current_time, min(window_end, end_time)))
            current_time = window_end

        return intervals

    def process_logs(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        node_num = len(self.service2node_id)
        event_num = len(self.log_templates) + 1

        template2id = {
            template: idx + 1 for idx, template in enumerate(self.log_templates)
        }
        template2id["UNSEEN"] = 0

        result = np.zeros((len(intervals), node_num, event_num))

        for file_type in ["normal_log", "abnormal_log"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=8)

            for chunk_idx, (start_time, end_time) in enumerate(
                tqdm(intervals, desc=f"Processing {file_type}")
            ):
                chunk_df = df[
                    (df["time"] >= start_time)
                    & (df["time"] <= end_time + pd.Timedelta(minutes=1))
                ]

                for service_name, service_group in chunk_df.groupby("service_name"):
                    if service_name not in self.service2node_id:
                        continue

                    node_id = self.service2node_id[service_name]

                    for message in service_group["message"]:
                        template_id = template2id.get(str(message), 0)
                        result[chunk_idx, node_id, template_id] += 1

        return result

    def process_metrics(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        node_num = len(self.service2node_id)
        metric_num = len(self.metric_names)

        result = np.zeros((len(intervals), node_num, self.chunk_length, metric_num))

        for file_type in ["normal_metric", "abnormal_metric"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])
            df["time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=8)
            df["service_name"] = df["attr.k8s.deployment.name"].fillna(
                df["attr.k8s.replicaset.name"]
            )
            for chunk_idx, (start_time, end_time) in enumerate(
                tqdm(intervals, desc=f"Processing {file_type}")
            ):
                chunk_df = df[(df["time"] >= start_time) & (df["time"] < end_time)]

                for (service_name, metric_name), group in chunk_df.groupby(
                    ["service_name", "metric"]
                ):
                    if (
                        service_name not in self.service2node_id
                        or metric_name not in self.metric_names
                    ):
                        continue

                    node_id = self.service2node_id[service_name]
                    metric_id = self.metric_names.index(metric_name)

                    group_sorted = group.sort_values("time")
                    values = group_sorted["value"].values

                    if len(values) > self.chunk_length:
                        indices = np.linspace(
                            0, len(values) - 1, self.chunk_length, dtype=int
                        )
                        values = values[indices]
                    elif len(values) < self.chunk_length:
                        padded_values = np.zeros(self.chunk_length)
                        padded_values[: len(values)] = values
                        values = padded_values

                    result[chunk_idx, node_id, :, metric_id] = values

        return result

    def process_traces(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        node_num = len(self.service2node_id)

        result = np.zeros((len(intervals), node_num, self.chunk_length, 1))

        for file_type in ["normal_trace", "abnormal_trace"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])

            df["time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=8)

            for chunk_idx, (start_time, end_time) in enumerate(
                tqdm(intervals, desc=f"Processing {file_type}")
            ):
                chunk_df = df[(df["time"] >= start_time) & (df["time"] < end_time)]

                for service_name, service_group in chunk_df.groupby("service_name"):
                    if service_name not in self.service2node_id:
                        continue

                    node_id = self.service2node_id[service_name]

                    durations = service_group["duration"].values
                    if len(durations) == 0:
                        continue

                    service_group_sorted = service_group.sort_values("time")
                    durations = service_group_sorted["duration"].values

                    if len(durations) > self.chunk_length:
                        indices = np.linspace(
                            0, len(durations) - 1, self.chunk_length, dtype=int
                        )
                        durations = durations[indices]
                    elif len(durations) < self.chunk_length:
                        padded_durations = np.zeros(self.chunk_length)
                        padded_durations[: len(durations)] = durations
                        durations = padded_durations

                    result[chunk_idx, node_id, :, 0] = durations

        return result

    def build_service_graph(self, data_files: Dict[str, Path]):
        edges = []
        edges_set = set()

        span_to_service = {}

        for file_type in ["normal_trace", "abnormal_trace"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])

            for _, row in df.iterrows():
                span_id = row.get("span_id")
                service_name = row.get("service_name")
                if pd.notna(span_id) and pd.notna(service_name):
                    span_to_service[span_id] = service_name

        for file_type in ["normal_trace", "abnormal_trace"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])

            for _, row in tqdm(df.iterrows(), desc=f"Building graph from {file_type}"):
                parent_span_id = row.get("parent_span_id")
                current_service = row.get("service_name")

                if (
                    pd.notna(parent_span_id)
                    and pd.notna(current_service)
                    and parent_span_id in span_to_service
                ):
                    parent_service = span_to_service[parent_span_id]

                    if (
                        parent_service != current_service
                        and parent_service in self.service2node_id
                        and current_service in self.service2node_id
                    ):
                        parent_node_id = self.service2node_id[parent_service]
                        current_node_id = self.service2node_id[current_service]

                        edge = (parent_node_id, current_node_id)
                        if edge not in edges_set:
                            edges.append(edge)
                            edges_set.add(edge)

        print(f"Built service graph with {len(edges)} edges")
        return edges

    def generate_fault_labels(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> List[int]:
        labels = []
        with open(data_files["injection"]) as f:
            injection_data = json.load(f)

        conf = json.loads(injection_data["display_config"])
        service = conf["injection_point"]["source_service"]

        # Convert injection times to UTC-aware timestamps for comparison
        injection_start = pd.to_datetime(injection_data["start_time"]).tz_localize(
            "UTC"
        )
        injection_end = pd.to_datetime(injection_data["end_time"]).tz_localize("UTC")

        for start_time, end_time in intervals:
            has_fault = False

            if injection_start < end_time and injection_end > start_time:
                has_fault = True

            if has_fault:
                labels.append(self.service2node_id[service])
            else:
                labels.append(-1)

        return labels

    def process_data_pack(self, data_pack_path: Path, output_dir: Path) -> None:
        print(f"Processing data pack: {data_pack_path}")

        data_files = self.derive_filename(data_pack_path)

        self.build_service_mapping(data_files)
        self.extract_log_templates(data_files)
        self.extract_metric_names(data_files)

        intervals = self.get_time_intervals(data_files)
        print(f"Generated {len(intervals)} time intervals")

        if len(intervals) == 0:
            print("No valid time intervals found, skipping...")
            return

        logs_data = self.process_logs(data_files, intervals)
        metrics_data = self.process_metrics(data_files, intervals)
        traces_data = self.process_traces(data_files, intervals)
        edges = self.build_service_graph(data_files)
        labels = self.generate_fault_labels(data_files, intervals)

        chunks = {}
        for i in range(len(intervals)):
            chunk_id = f"chunk_{i:06d}"
            chunks[chunk_id] = {
                "logs": logs_data[i],
                "metrics": metrics_data[i],
                "traces": traces_data[i],
                "culprit": labels[i],
            }

        chunk_ids = list(chunks.keys())
        np.random.shuffle(chunk_ids)

        split_idx = int(len(chunk_ids) * 0.7)
        train_ids = chunk_ids[:split_idx]
        test_ids = chunk_ids[split_idx:]

        train_chunks = {cid: chunks[cid] for cid in train_ids}
        test_chunks = {cid: chunks[cid] for cid in test_ids}

        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "chunk_train.pkl", "wb") as f:
            pickle.dump(train_chunks, f)

        with open(output_dir / "chunk_test.pkl", "wb") as f:
            pickle.dump(test_chunks, f)

        metadata = {
            "event_num": len(self.log_templates) + 1,
            "metric_num": len(self.metric_names),
            "node_num": len(self.service2node_id),
            "chunk_length": self.chunk_length,
            "service2node_id": self.service2node_id,
            "node_id2service": self.node_id2service,
            "log_templates": self.log_templates,
            "metric_names": self.metric_names,
            "edges": edges,
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Data processing completed. Output saved to: {output_dir}")
        print(f"Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}")


def main():
    data_root = Path("sdata")
    output_root = Path("data")

    # cases = pd.read_parquet(
    #     "/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet"
    # )
    # top_10 = cases["datapack"].head(10).tolist()
    top_10 = ["ts5-ts-route-service-partition-bbphlf"]
    adapter = DataAdapter(chunk_length=1)

    for data_pack_name in top_10:
        data_pack_path = data_root / data_pack_name
        output_path = output_root / data_pack_name

        try:
            adapter.process_data_pack(data_pack_path, output_path)
        except Exception as e:
            print(f"Error processing {data_pack_name}: {str(e)}")
            raise e


if __name__ == "__main__":
    main()
