import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from .utils import CacheManager
from .progress_manager import TaskStatus
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


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
        self.all_edges = set()  # 用于收集所有边
        self.global_services = set()  # 用于收集所有服务
        self.global_log_messages = []  # 用于收集所有日志消息
        self.global_metrics = set()  # 用于收集所有指标

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
                    messages = df["message"].astype(str).tolist()
                    print(f"Processing {len(messages)} messages from {file_type}")
                    processed_messages = [
                        self.drain(msg) for msg in messages if msg.strip()
                    ]
                    all_messages.extend(processed_messages)
                else:
                    print(f"Warning: 'message' column not found in {file_type}")
            else:
                print(f"Warning: {file_type} file does not exist")

        message_counts = Counter(all_messages)
        valid_templates = [
            (msg, count) for msg, count in message_counts.items() if msg.strip()
        ]
        self.log_templates = [
            msg for msg, count in Counter(dict(valid_templates)).most_common(100)
        ]

        print(f"Extracted {len(self.log_templates)} log templates")
        if len(self.log_templates) == 0:
            print("Warning: No log templates extracted!")

        self.drain.save_cache()

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
            window_end = current_time + pd.Timedelta(minutes=2)
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
                        # 使用 drain 处理后的模板进行匹配
                        template = self.drain(str(message))
                        template_id = template2id.get(template, 0)
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

                    # 确保数据长度匹配 chunk_length
                    if len(values) > self.chunk_length:
                        # 如果数据点太多，进行均匀采样
                        indices = np.linspace(
                            0, len(values) - 1, self.chunk_length, dtype=int
                        )
                        values = values[indices]
                    elif len(values) < self.chunk_length:
                        # 如果数据点不足，进行零填充
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

                    # 确保数据长度匹配 chunk_length
                    if len(durations) > self.chunk_length:
                        # 如果数据点太多，进行均匀采样
                        indices = np.linspace(
                            0, len(durations) - 1, self.chunk_length, dtype=int
                        )
                        durations = durations[indices]
                    elif len(durations) < self.chunk_length:
                        # 如果数据点不足，进行零填充
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
        service = conf["injection_point"].get("source_service", "")
        if service == "":
            service = conf["injection_point"].get("app_name", "")
        if service == "":
            service = conf["injection_point"].get("app_label", "")
        assert service != ""

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

        # 验证数据维度
        if train_chunks:
            sample_chunk = next(iter(train_chunks.values()))
            print("Sample chunk data shapes:")
            print(f"  Logs: {sample_chunk['logs'].shape}")
            print(f"  Metrics: {sample_chunk['metrics'].shape}")
            print(f"  Traces: {sample_chunk['traces'].shape}")
            print(f"  Culprit: {sample_chunk['culprit']}")

        print("Metadata summary:")
        print(f"  Event num: {len(self.log_templates) + 1}")
        print(f"  Metric num: {len(self.metric_names)}")
        print(f"  Node num: {len(self.service2node_id)}")
        print(f"  Edges: {len(edges)}")
        print(f"  Chunk length: {self.chunk_length}")

    def collect_global_metadata_parallel(
        self, data_packs: List[Path], n_workers: Optional[int] = None
    ) -> None:
        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(data_packs))

        print(
            f"Collecting metadata from {len(data_packs)} cases using {n_workers} workers..."
        )

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(collect_metadata_worker, data_pack)
                for data_pack in data_packs
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Collecting metadata"
            ):
                try:
                    metadata = future.result()
                    self.global_services.update(metadata["services"])
                    self.global_metrics.update(metadata["metrics"])
                    self.global_log_messages.extend(metadata["log_messages"])
                except Exception as e:
                    print(f"Error processing metadata result: {str(e)}")

        print(f"Collected metadata from {len(data_packs)} cases")
        print(f"  Total services: {len(self.global_services)}")
        print(f"  Total metrics: {len(self.global_metrics)}")
        print(f"  Total log messages: {len(self.global_log_messages)}")

    def collect_global_metadata(self, data_files: Dict[str, Path]) -> None:
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
                    self.global_services.update(
                        df[df["service_name"].notna()]["service_name"].unique()
                    )

        for file_type in ["normal_metric", "abnormal_metric"]:
            if data_files[file_type].exists():
                df = pd.read_parquet(data_files[file_type])
                if "metric" in df.columns:
                    self.global_metrics.update(df["metric"].unique())

        for file_type in ["normal_log", "abnormal_log"]:
            if data_files[file_type].exists():
                df = pd.read_parquet(data_files[file_type])
                if "message" in df.columns:
                    messages = df["message"].astype(str).tolist()
                    valid_messages = [msg for msg in messages if msg.strip()]
                    self.global_log_messages.extend(valid_messages)

    def build_global_mappings(self) -> None:
        print(f"Building global mappings from {len(self.global_services)} services")

        self.service2node_id = {
            service: idx for idx, service in enumerate(sorted(self.global_services))
        }
        self.node_id2service = {
            idx: service for service, idx in self.service2node_id.items()
        }

        self.metric_names = sorted(list(self.global_metrics))

        print(
            f"Processing {len(self.global_log_messages)} log messages for template extraction"
        )
        processed_messages = []
        batch_size = 1000
        for i in range(0, len(self.global_log_messages), batch_size):
            batch = self.global_log_messages[i : i + batch_size]
            batch_processed = [self.drain(msg) for msg in batch if msg.strip()]
            processed_messages.extend(batch_processed)

            if i % (batch_size * 10) == 0:
                print(f"Processed {i + len(batch)} messages...")

        message_counts = Counter(processed_messages)
        valid_templates = [
            (msg, count) for msg, count in message_counts.items() if msg.strip()
        ]
        self.log_templates = [
            msg for msg, count in Counter(dict(valid_templates)).most_common(100)
        ]

        self.global_log_messages.clear()

        print("Global mappings built:")
        print(f"  Services: {len(self.service2node_id)}")
        print(f"  Metrics: {len(self.metric_names)}")
        print(f"  Log templates: {len(self.log_templates)}")

        self.drain.save_cache()

    def process_cases_parallel(
        self, data_packs: List[Path], n_workers: Optional[int] = None
    ) -> Dict:
        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(data_packs))

        print(f"Processing {len(data_packs)} cases using {n_workers} workers...")

        drain_config = {
            "service2node_id": self.service2node_id,
            "node_id2service": self.node_id2service,
            "log_templates": self.log_templates,
            "metric_names": self.metric_names,
        }

        work_args = [
            (data_pack, self.chunk_length, drain_config) for data_pack in data_packs
        ]

        all_chunks = {}

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_single_case_worker, args) for args in work_args
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing cases"
            ):
                try:
                    case_name, chunks, edges = future.result()
                    all_chunks.update(chunks)
                    self.all_edges.update(edges)
                    print(f"Processed {case_name}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"Error processing case result: {str(e)}")

        print(f"Parallel processing completed. Total chunks: {len(all_chunks)}")
        return all_chunks

    def process_single_case(self, data_pack_path: Path) -> Dict:
        print(f"Processing single case: {data_pack_path}")

        data_files = self.derive_filename(data_pack_path)
        intervals = self.get_time_intervals(data_files)

        if len(intervals) == 0:
            print("No valid time intervals found, skipping...")
            return {}

        logs_data = self.process_logs(data_files, intervals)
        metrics_data = self.process_metrics(data_files, intervals)
        traces_data = self.process_traces(data_files, intervals)
        edges = self.build_service_graph(data_files)
        labels = self.generate_fault_labels(data_files, intervals)

        self.all_edges.update(edges)

        chunks = {}
        for i in range(len(intervals)):
            chunk_id = f"{data_pack_path.name}_chunk_{i:06d}"
            chunks[chunk_id] = {
                "logs": logs_data[i],
                "metrics": metrics_data[i],
                "traces": traces_data[i],
                "culprit": labels[i],
            }

        print(f"Generated {len(chunks)} chunks from {data_pack_path.name}")
        return chunks

    def save_dataset_batch(
        self, all_chunks: Dict, output_dir: Path, train_ratio: float = 0.7
    ) -> None:
        """保存整个数据集"""
        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_ids = list(all_chunks.keys())
        np.random.shuffle(chunk_ids)

        split_idx = int(len(chunk_ids) * train_ratio)
        train_ids = chunk_ids[:split_idx]
        test_ids = chunk_ids[split_idx:]

        train_chunks = {cid: all_chunks[cid] for cid in train_ids}
        test_chunks = {cid: all_chunks[cid] for cid in test_ids}

        print(
            f"Saving dataset with {len(train_chunks)} train and {len(test_chunks)} test chunks"
        )

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
            "edges": list(self.all_edges),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Dataset saved to: {output_dir}")
        print(f"Final dataset statistics:")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Train chunks: {len(train_chunks)}")
        print(f"  Test chunks: {len(test_chunks)}")
        print(f"  Services: {len(self.service2node_id)}")
        print(f"  Metrics: {len(self.metric_names)}")
        print(f"  Log templates: {len(self.log_templates)}")
        print(f"  Edges: {len(self.all_edges)}")


def create_dataset_streaming(
    data_root: str,
    cases_file: str,
    output_dir: str,
    max_cases: Optional[int] = None,
    batch_size: int = 2,
    chunk_length: int = 10,
    train_ratio: float = 0.7,
    n_workers: Optional[int] = None,
    use_parallel: bool = True,
    progress_manager=None,
):
    data_root_path = Path(data_root)
    output_root = Path(output_dir)

    cases = pd.read_parquet(cases_file)
    if max_cases:
        cases_list = cases["datapack"].head(max_cases).tolist()
    else:
        cases_list = cases["datapack"].tolist()

    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(cases_list))

    adapter = DataAdapter(chunk_length=chunk_length)
    data_packs = [data_root_path / name for name in cases_list]

    if use_parallel:
        if progress_manager:
            progress_manager.add_task(
                "metadata", "Collecting Metadata", len(data_packs)
            )
            progress_manager.add_task("mapping", "Building Mappings", 1)
            progress_manager.add_task("processing", "Processing Cases", len(data_packs))
            progress_manager.add_task("saving", "Saving Dataset", 1)

            for i, data_pack in enumerate(data_packs):
                progress_manager.add_task(
                    f"meta_{i}", f"Meta: {data_pack.name[:20]}..."
                )
                progress_manager.add_task(
                    f"proc_{i}", f"Proc: {data_pack.name[:20]}..."
                )

        if progress_manager:
            progress_manager.update_task(
                "metadata",
                status=TaskStatus.RUNNING,
                current_step="Starting metadata collection",
            )

        metadata_results = collect_global_metadata_parallel_with_progress(
            data_packs, n_workers, progress_manager
        )

        for metadata in metadata_results:
            adapter.global_services.update(metadata["services"])
            adapter.global_metrics.update(metadata["metrics"])
            adapter.global_log_messages.extend(metadata["log_messages"])

        if progress_manager:
            progress_manager.update_task(
                "metadata", status=TaskStatus.COMPLETED, progress=100.0
            )

        # Phase 2: 构建映射
        if progress_manager:
            progress_manager.update_task(
                "mapping",
                status=TaskStatus.RUNNING,
                current_step="Building global mappings",
            )

        adapter.build_global_mappings()

        if progress_manager:
            progress_manager.update_task(
                "mapping", status=TaskStatus.COMPLETED, progress=100.0
            )

        if progress_manager:
            progress_manager.update_task(
                "processing",
                status=TaskStatus.RUNNING,
                current_step="Processing all cases",
            )

        all_chunks = process_cases_parallel_with_progress(
            data_packs, adapter, n_workers, progress_manager
        )

        if progress_manager:
            progress_manager.update_task(
                "processing", status=TaskStatus.COMPLETED, progress=100.0
            )

    else:
        if progress_manager:
            progress_manager.add_task(
                "sequential", "Sequential Processing", len(cases_list)
            )
            progress_manager.update_task("sequential", status=TaskStatus.RUNNING)

        for i, data_pack_name in enumerate(cases_list):
            data_pack_path = data_root_path / data_pack_name

            if progress_manager:
                progress_manager.update_task(
                    "sequential",
                    progress=(i / len(cases_list)) * 25,  # 25% for metadata
                    current_step=f"Collecting metadata: {data_pack_name[:30]}...",
                )

            try:
                data_files = adapter.derive_filename(data_pack_path)
                adapter.collect_global_metadata(data_files)
            except Exception as e:
                if progress_manager:
                    progress_manager.update_task(
                        "sequential",
                        current_step=f"Error in {data_pack_name}: {str(e)[:50]}...",
                    )
                continue

        if progress_manager:
            progress_manager.update_task(
                "sequential", progress=25, current_step="Building global mappings..."
            )
        adapter.build_global_mappings()

        all_chunks = {}
        processed_count = 0

        for i in range(0, len(cases_list), batch_size):
            batch_cases = cases_list[i : i + batch_size]

            for data_pack_name in batch_cases:
                data_pack_path = data_root_path / data_pack_name

                if progress_manager:
                    progress_manager.update_task(
                        "sequential",
                        progress=25 + (processed_count / len(cases_list)) * 70,
                        current_step=f"Processing: {data_pack_name[:30]}...",
                    )

                try:
                    case_chunks = adapter.process_single_case(data_pack_path)
                    all_chunks.update(case_chunks)
                except Exception as e:
                    if progress_manager:
                        progress_manager.update_task(
                            "sequential",
                            current_step=f"Error processing {data_pack_name}: {str(e)[:50]}...",
                        )
                    continue
                finally:
                    processed_count += 1

        if progress_manager:
            progress_manager.update_task(
                "sequential", status=TaskStatus.COMPLETED, progress=100.0
            )

    if progress_manager:
        if use_parallel:
            progress_manager.update_task(
                "saving", status=TaskStatus.RUNNING, current_step="Saving dataset..."
            )
        else:
            progress_manager.update_task("sequential", current_step="Saving dataset...")

    adapter.save_dataset_batch(all_chunks, output_root, train_ratio)

    if progress_manager:
        if use_parallel:
            progress_manager.update_task(
                "saving", status=TaskStatus.COMPLETED, progress=100.0
            )

    return len(all_chunks)


def process_single_case_worker(args):
    data_pack_path, chunk_length, drain_config = args

    try:
        adapter = DataAdapter(chunk_length=chunk_length)

        adapter.service2node_id = drain_config["service2node_id"]
        adapter.node_id2service = drain_config["node_id2service"]
        adapter.log_templates = drain_config["log_templates"]
        adapter.metric_names = drain_config["metric_names"]

        chunks = adapter.process_single_case(data_pack_path)

        data_files = adapter.derive_filename(data_pack_path)
        edges = adapter.build_service_graph(data_files)

        return data_pack_path.name, chunks, edges

    except Exception as e:
        print(f"Error processing {data_pack_path.name} in worker: {str(e)}")
        return data_pack_path.name, {}, []


def collect_metadata_worker(data_pack_path):
    try:
        adapter = DataAdapter()
        data_files = adapter.derive_filename(data_pack_path)

        services = set()
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
                    services.update(
                        df[df["service_name"].notna()]["service_name"].unique()
                    )

        metrics = set()
        for file_type in ["normal_metric", "abnormal_metric"]:
            if data_files[file_type].exists():
                df = pd.read_parquet(data_files[file_type])
                if "metric" in df.columns:
                    metrics.update(df["metric"].unique())

        log_messages = []
        for file_type in ["normal_log", "abnormal_log"]:
            if data_files[file_type].exists():
                df = pd.read_parquet(data_files[file_type])
                if "message" in df.columns:
                    messages = df["message"].astype(str).tolist()
                    valid_messages = [msg for msg in messages if msg.strip()]
                    log_messages.extend(valid_messages[:500])  # 限制每个文件的消息数量

        return {
            "case_name": data_pack_path.name,
            "services": services,
            "metrics": metrics,
            "log_messages": log_messages,
        }

    except Exception as e:
        print(f"Error collecting metadata from {data_pack_path.name}: {str(e)}")
        return {
            "case_name": data_pack_path.name,
            "services": set(),
            "metrics": set(),
            "log_messages": [],
        }


def collect_global_metadata_parallel_with_progress(
    data_packs: List[Path], n_workers: int, progress_manager=None
) -> List[Dict]:
    metadata_results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(collect_metadata_worker, data_pack): i
            for i, data_pack in enumerate(data_packs)
        }

        # 收集结果
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                metadata = future.result()
                metadata_results.append(metadata)

                # 更新进度
                if progress_manager:
                    progress = (len(metadata_results) / len(data_packs)) * 100
                    progress_manager.update_task(
                        f"meta_{index}",
                        status=TaskStatus.COMPLETED,
                        progress=100.0,
                        current_step="Metadata collected",
                    )
                    progress_manager.update_task(
                        "metadata",
                        progress=progress,
                        current_step=f"Completed {len(metadata_results)}/{len(data_packs)} cases",
                    )

            except Exception as e:
                if progress_manager:
                    progress_manager.update_task(
                        f"meta_{index}", status=TaskStatus.FAILED, error_msg=str(e)
                    )

    return metadata_results


def process_cases_parallel_with_progress(
    data_packs: List[Path], adapter: DataAdapter, n_workers: int, progress_manager=None
) -> Dict:
    drain_config = {
        "service2node_id": adapter.service2node_id,
        "node_id2service": adapter.node_id2service,
        "log_templates": adapter.log_templates,
        "metric_names": adapter.metric_names,
    }

    work_args = [
        (data_pack, adapter.chunk_length, drain_config) for data_pack in data_packs
    ]

    all_chunks = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(process_single_case_worker, args): i
            for i, args in enumerate(work_args)
        }

        completed_count = 0

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                case_name, chunks, edges = future.result()
                all_chunks.update(chunks)
                adapter.all_edges.update(edges)
                completed_count += 1

                if progress_manager:
                    progress = (completed_count / len(data_packs)) * 100
                    progress_manager.update_task(
                        f"proc_{index}",
                        status=TaskStatus.COMPLETED,
                        progress=100.0,
                        current_step=f"Generated {len(chunks)} chunks",
                    )
                    progress_manager.update_task(
                        "processing",
                        progress=progress,
                        current_step=f"Completed {completed_count}/{len(data_packs)} cases",
                    )

            except Exception as e:
                if progress_manager:
                    progress_manager.update_task(
                        f"proc_{index}", status=TaskStatus.FAILED, error_msg=str(e)
                    )
                completed_count += 1

    return all_chunks
