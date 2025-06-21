import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from .utils import CacheManager, timeit
from .progress_manager import TaskStatus
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed


class GlobalMetadata:
    def __init__(self):
        self.services: Set[str] = set()
        self.metrics: Set[str] = set()
        self.log_templates: List[str] = []
        self.service2node_id: Dict[str, int] = {}
        self.node_id2service: Dict[int, str] = {}
        self.metric_names: List[str] = []
        self.template2id: Dict[str, int] = {}
        self.all_edges: Set[Tuple[int, int]] = set()

    def update_from_case(self, case_metadata: Dict):
        self.services.update(case_metadata.get("services", set()))
        self.metrics.update(case_metadata.get("metrics", set()))

    def finalize_mappings(self, log_templates: List[str]):
        self.service2node_id = {
            service: idx for idx, service in enumerate(sorted(self.services))
        }
        self.node_id2service = {
            idx: service for service, idx in self.service2node_id.items()
        }

        # Build metric mappings
        self.metric_names = sorted(list(self.metrics))

        # Build log template mappings
        self.log_templates = log_templates
        self.template2id = {
            template: idx + 1 for idx, template in enumerate(self.log_templates)
        }
        self.template2id["UNSEEN"] = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "service2node_id": self.service2node_id,
            "node_id2service": self.node_id2service,
            "log_templates": self.log_templates,
            "metric_names": self.metric_names,
            "template2id": self.template2id,
            "edges": list(self.all_edges),
            "node_num": len(self.service2node_id),
            "event_num": len(self.log_templates) + 1,
            "metric_num": len(self.metric_names),
        }


class DrainProcessor:
    def __init__(self, conf: str, save_path: str, cache_dir: str = "./cache/drain"):
        persistence = FilePersistence(save_path)
        miner_config = TemplateMinerConfig()
        miner_config.load(conf)
        self._template_miner = TemplateMiner(persistence, config=miner_config)
        self._cache_manager = CacheManager[str](
            Path(cache_dir) / "sentence_templates.pkl"
        )

    def process(self, sentence: str) -> str:
        line = str(sentence).strip()
        if not line:
            return ""

        return self._cache_manager.get_or_compute(
            line, lambda: self._extract_template(line)
        )

    def _extract_template(self, line: str) -> str:
        result = self._template_miner.add_log_message(line)
        template = result.get("template_mined")
        if template is None:
            logging.warning(f"Failed to extract template for: {line}")
            return ""
        return template

    def save_cache(self):
        self._cache_manager.save()


class CaseProcessor:
    def __init__(self, chunk_length: int, global_metadata: GlobalMetadata):
        self.chunk_length = chunk_length
        self.global_metadata = global_metadata

    def derive_filenames(self, data_pack: Path) -> Dict[str, Path]:
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

    @timeit()
    def process_logs(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        node_num = len(self.global_metadata.service2node_id)
        event_num = len(self.global_metadata.log_templates) + 1
        result = np.zeros((len(intervals), node_num, event_num))

        for file_type in ["normal_log", "abnormal_log"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=8)

            # 预处理：过滤有效服务和映射模板ID
            df = df[df["service_name"].isin(self.global_metadata.service2node_id)]
            if df.empty:
                continue

            # 向量化映射service到node_id
            df["node_id"] = df["service_name"].map(self.global_metadata.service2node_id)

            # 向量化映射message到template_id
            df["message_str"] = df["message"].astype(str).str.strip()
            df["template_id"] = (
                df["message_str"]
                .map(self.global_metadata.template2id)
                .fillna(0)
                .astype(int)
            )

            # 为每个时间段分配chunk_idx
            for chunk_idx, (start_time, end_time) in enumerate(intervals):
                chunk_mask = (df["time"] >= start_time) & (
                    df["time"] <= end_time + pd.Timedelta(minutes=1)
                )
                chunk_df = df[chunk_mask]

                if chunk_df.empty:
                    continue

                # 使用groupby和size()进行向量化计数
                counts = chunk_df.groupby(["node_id", "template_id"]).size()

                # 批量更新结果矩阵
                for (node_id, template_id), count in counts.items():
                    result[chunk_idx, node_id, template_id] += count

        return result

    @timeit()
    def process_metrics(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        node_num = len(self.global_metadata.service2node_id)
        metric_num = len(self.global_metadata.metric_names)
        result = np.zeros((len(intervals), node_num, self.chunk_length, metric_num))

        for file_type in ["normal_metric", "abnormal_metric"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])
            df["time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=8)
            df["service_name"] = df["attr.k8s.deployment.name"].fillna(
                df["attr.k8s.replicaset.name"]
            )

            # 预过滤有效的服务和指标
            valid_services_mask = df["service_name"].isin(
                self.global_metadata.service2node_id
            )
            valid_metrics_mask = df["metric"].isin(self.global_metadata.metric_names)
            df = df[valid_services_mask & valid_metrics_mask]

            if df.empty:
                continue

            # 向量化映射
            df["node_id"] = df["service_name"].map(self.global_metadata.service2node_id)
            df["metric_id"] = df["metric"].map(
                {
                    metric: idx
                    for idx, metric in enumerate(self.global_metadata.metric_names)
                }
            )

            # 使用searchsorted进行向量化时间段分配
            interval_starts = pd.Series([start for start, _ in intervals])
            interval_ends = pd.Series([end for _, end in intervals])

            # 为每个记录分配时间段索引
            df["chunk_idx"] = pd.cut(
                df["time"],
                bins=pd.concat(
                    [interval_starts, interval_ends.iloc[-1:]], ignore_index=True
                ),
                labels=range(len(intervals)),
                include_lowest=True,
                right=False,
            )

            # 移除未分配到任何时间段的记录
            df = df.dropna(subset=["chunk_idx"])
            df["chunk_idx"] = df["chunk_idx"].astype(int)

            # 按(chunk_idx, node_id, metric_id)分组处理
            for (chunk_idx, node_id, metric_id), group in df.groupby(
                ["chunk_idx", "node_id", "metric_id"]
            ):
                values = group.sort_values("time")["value"].values
                values = self._normalize_sequence_length(
                    np.array(values), self.chunk_length
                )
                result[chunk_idx, node_id, :, metric_id] = values

        return result

    @timeit()
    def process_traces(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        """Process trace data using global metadata mappings"""
        node_num = len(self.global_metadata.service2node_id)
        result = np.zeros((len(intervals), node_num, self.chunk_length, 1))

        for file_type in ["normal_trace", "abnormal_trace"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])
            df["time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=8)

            # 预过滤有效的服务
            df = df[df["service_name"].isin(self.global_metadata.service2node_id)]
            if df.empty:
                continue

            # 向量化映射
            df["node_id"] = df["service_name"].map(self.global_metadata.service2node_id)

            # 为每个记录分配时间段索引
            interval_starts = pd.Series([start for start, _ in intervals])
            interval_ends = pd.Series([end for _, end in intervals])

            df["chunk_idx"] = pd.cut(
                df["time"],
                bins=pd.concat(
                    [interval_starts, interval_ends.iloc[-1:]], ignore_index=True
                ),
                labels=range(len(intervals)),
                include_lowest=True,
                right=False,
            )

            # 移除未分配到任何时间段的记录
            df = df.dropna(subset=["chunk_idx"])
            df["chunk_idx"] = df["chunk_idx"].astype(int)

            # 按(chunk_idx, node_id)分组处理
            for (chunk_idx, node_id), group in df.groupby(["chunk_idx", "node_id"]):
                durations = group.sort_values("time")["duration"].values
                if len(durations) > 0:
                    durations = self._normalize_sequence_length(
                        np.array(durations), self.chunk_length
                    )
                    result[chunk_idx, node_id, :, 0] = durations

        return result

    @timeit()
    def extract_service_graph(
        self, data_files: Dict[str, Path]
    ) -> Set[Tuple[int, int]]:
        edges = set()
        span_to_service = {}

        # 首先构建 span_id 到 service_name 的映射
        for file_type in ["normal_trace", "abnormal_trace"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])
            valid_rows = df[["span_id", "service_name"]].dropna()
            if not valid_rows.empty:
                span_to_service.update(
                    dict(zip(valid_rows["span_id"], valid_rows["service_name"]))
                )

        # 然后提取服务间的边关系
        for file_type in ["normal_trace", "abnormal_trace"]:
            if not data_files[file_type].exists():
                continue

            df = pd.read_parquet(data_files[file_type])
            valid_df = df[["parent_span_id", "service_name"]].dropna()
            if valid_df.empty:
                continue

            # 向量化过滤有效的parent_span_id
            valid_df = valid_df[valid_df["parent_span_id"].isin(span_to_service)]
            if valid_df.empty:
                continue

            # 向量化映射parent_service
            valid_df = valid_df.copy()
            valid_df["parent_service"] = valid_df["parent_span_id"].map(span_to_service)

            # 过滤自循环
            valid_df = valid_df[valid_df["parent_service"] != valid_df["service_name"]]

            # 过滤有效服务
            valid_services = set(self.global_metadata.service2node_id.keys())
            valid_df = valid_df[
                valid_df["parent_service"].isin(valid_services)
                & valid_df["service_name"].isin(valid_services)
            ]

            if not valid_df.empty:
                # 向量化映射到node_id
                parent_ids = valid_df["parent_service"].map(
                    self.global_metadata.service2node_id
                )
                current_ids = valid_df["service_name"].map(
                    self.global_metadata.service2node_id
                )

                # 批量添加边
                new_edges = set(zip(parent_ids, current_ids))
                edges.update(new_edges)

        return edges

    @timeit()
    def generate_fault_labels(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> List[int]:
        with open(data_files["injection"]) as f:
            injection_data = json.load(f)

        conf = json.loads(injection_data["display_config"])
        service = (
            conf["injection_point"].get("source_service", "")
            or conf["injection_point"].get("app_name", "")
            or conf["injection_point"].get("app_label", "")
        )

        if not service:
            raise ValueError("No fault service found in injection config")

        injection_start = pd.to_datetime(injection_data["start_time"]).tz_localize(
            "UTC"
        )
        injection_end = pd.to_datetime(injection_data["end_time"]).tz_localize("UTC")

        labels = []
        for start_time, end_time in intervals:
            if injection_start < end_time and injection_end > start_time:
                labels.append(self.global_metadata.service2node_id.get(service, -1))
            else:
                labels.append(-1)

        return labels

    def process_case(self, data_pack_path: Path) -> Dict:
        data_files = self.derive_filenames(data_pack_path)
        intervals = self.get_time_intervals(data_files)

        if not intervals:
            return {}

        logs_data = self.process_logs(data_files, intervals)
        metrics_data = self.process_metrics(data_files, intervals)
        traces_data = self.process_traces(data_files, intervals)
        edges = self.extract_service_graph(data_files)
        labels = self.generate_fault_labels(data_files, intervals)

        chunks = {}
        for i in range(len(intervals)):
            chunk_id = f"{data_pack_path.name}_chunk_{i:06d}"
            chunks[chunk_id] = {
                "logs": logs_data[i],
                "metrics": metrics_data[i],
                "traces": traces_data[i],
                "culprit": labels[i],
            }

        return {"chunks": chunks, "edges": edges}

    @staticmethod
    def _normalize_sequence_length(
        values: np.ndarray, target_length: int
    ) -> np.ndarray:
        if len(values) > target_length:
            indices = np.linspace(0, len(values) - 1, target_length, dtype=int)
            return values[indices]
        elif len(values) < target_length:
            padded = np.zeros(target_length)
            padded[: len(values)] = values
            return padded
        return values


class DatasetBuilder:
    """Main class for building datasets with proper global metadata handling"""

    def __init__(self, chunk_length: int = 10):
        self.chunk_length = chunk_length
        self.global_metadata = GlobalMetadata()

    def collect_metadata_from_case(self, data_pack_path: Path) -> Dict:
        """Collect metadata from a single case"""
        return collect_metadata_worker(data_pack_path, self.chunk_length)

    def build_global_metadata(
        self,
        data_packs: List[Path],
        output_dir: Path,
        n_workers: Optional[int] = None,
        enable_checkpointing: bool = True,
    ):
        """Build global metadata from all cases with checkpoint support"""

        # 尝试从检查点恢复
        if enable_checkpointing and self.load_checkpoint(output_dir):
            print("Resumed global metadata from checkpoint")
            return

        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(data_packs), 8)  # 限制最大线程数

        print(
            f"Collecting metadata from {len(data_packs)} cases using {n_workers} workers..."
        )

        # 使用分块策略处理以控制内存使用
        chunk_size = 50  # 每批处理50个案例
        all_log_messages = []

        for i in range(0, len(data_packs), chunk_size):
            batch_data_packs = data_packs[i : i + chunk_size]

            with ThreadPoolExecutor(
                max_workers=min(n_workers, len(batch_data_packs))
            ) as executor:
                futures = [
                    executor.submit(
                        collect_metadata_worker, data_pack, self.chunk_length
                    )
                    for data_pack in batch_data_packs
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Collecting metadata batch {i // chunk_size + 1}",
                ):
                    try:
                        metadata = future.result(timeout=300)  # 5分钟超时
                        self.global_metadata.update_from_case(metadata)
                        all_log_messages.extend(metadata["log_messages"])

                        # 控制内存使用：限制消息数量
                        if len(all_log_messages) > 200000:
                            # 随机采样保留200000条消息
                            import random

                            all_log_messages = random.sample(all_log_messages, 200000)

                    except Exception as e:
                        logging.error(f"Error processing metadata: {e}")

        print(f"Extracting log templates from {len(all_log_messages)} messages...")

        # 并行处理日志模板提取
        batch_size = max(1000, len(all_log_messages) // (n_workers * 2))
        message_batches = [
            all_log_messages[i : i + batch_size]
            for i in range(0, len(all_log_messages), batch_size)
        ]

        processed_messages = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_log_templates_worker, batch)
                for batch in message_batches
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing log templates",
            ):
                try:
                    batch_results = future.result(timeout=600)  # 10分钟超时
                    processed_messages.extend(batch_results)
                except Exception as e:
                    logging.error(f"Error processing log template batch: {e}")

        # 获取前100个最常见的模板
        template_counts = Counter(processed_messages)
        top_templates = [template for template, _ in template_counts.most_common(100)]

        self.global_metadata.finalize_mappings(top_templates)

        # 保存检查点
        if enable_checkpointing:
            self.save_checkpoint(output_dir)

        print("Global metadata built:")
        print(f"  Services: {len(self.global_metadata.services)}")
        print(f"  Metrics: {len(self.global_metadata.metrics)}")
        print(f"  Log templates: {len(self.global_metadata.log_templates)}")

        # 清理临时数据以释放内存
        del all_log_messages
        del processed_messages

    def process_cases_parallel(
        self,
        data_packs: List[Path],
        output_dir: Path,
        n_workers: Optional[int] = None,
        enable_checkpointing: bool = True,
    ) -> Dict:
        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(data_packs), 8)  # 限制最大线程数

        # 尝试从断点恢复
        all_chunks = {}
        all_edges = set()

        if enable_checkpointing:
            # 加载已有的中间结果
            all_chunks, all_edges, _ = self.load_intermediate_results(output_dir)

            # 获取剩余需要处理的案例
            remaining_data_packs = self.get_remaining_cases(data_packs, output_dir)

            if len(remaining_data_packs) < len(data_packs):
                print(
                    f"Resume from checkpoint: {len(data_packs) - len(remaining_data_packs)} cases already processed"
                )

            data_packs = remaining_data_packs

        if not data_packs:
            print("All cases already processed!")
            self.global_metadata.all_edges.update(all_edges)
            return all_chunks

        print(f"Processing {len(data_packs)} cases using {n_workers} workers...")

        metadata_dict = self.global_metadata.to_dict()

        # 批量处理以减少内存压力
        batch_size = 20  # 每批处理20个案例

        for i in range(0, len(data_packs), batch_size):
            batch_data_packs = data_packs[i : i + batch_size]

            with ThreadPoolExecutor(
                max_workers=min(n_workers, len(batch_data_packs))
            ) as executor:
                futures = [
                    executor.submit(
                        process_case_worker, data_pack, self.chunk_length, metadata_dict
                    )
                    for data_pack in batch_data_packs
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing batch {i // batch_size + 1}/{(len(data_packs) + batch_size - 1) // batch_size}",
                ):
                    try:
                        result = future.result(timeout=600)  # 10分钟超时
                        all_chunks.update(result["chunks"])
                        all_edges.update(result["edges"])

                        # 保存中间结果（如果启用检查点）
                        if enable_checkpointing:
                            case_name = (
                                list(result["chunks"].keys())[0].split("_chunk_")[0]
                                if result["chunks"]
                                else "unknown"
                            )
                            self.save_intermediate_result(
                                case_name, result["chunks"], result["edges"], output_dir
                            )

                    except Exception as e:
                        logging.error(f"Error processing case: {e}")

            # 每批次后保存检查点
            if enable_checkpointing:
                self.save_checkpoint(output_dir)

        self.global_metadata.all_edges.update(all_edges)
        return all_chunks

    def save_dataset(self, chunks: Dict, output_dir: Path, train_ratio: float = 0.7):
        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_ids = list(chunks.keys())
        np.random.shuffle(chunk_ids)
        split_idx = int(len(chunk_ids) * train_ratio)

        train_chunks = {cid: chunks[cid] for cid in chunk_ids[:split_idx]}
        test_chunks = {cid: chunks[cid] for cid in chunk_ids[split_idx:]}

        with open(output_dir / "chunk_train.pkl", "wb") as f:
            pickle.dump(train_chunks, f)
        with open(output_dir / "chunk_test.pkl", "wb") as f:
            pickle.dump(test_chunks, f)

        metadata = self.global_metadata.to_dict()
        metadata["chunk_length"] = self.chunk_length

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Dataset saved to {output_dir}")
        print(f"Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}")
        print(
            f"Metadata: {metadata['node_num']} nodes, {metadata['event_num']} events, {metadata['metric_num']} metrics"
        )

        # 数据集保存完成后，可以选择清理中间文件
        # self.cleanup_intermediate_files(output_dir)

    def save_intermediate_result(
        self, case_name: str, chunks: Dict, edges: Set, output_dir: Path
    ) -> None:
        """保存单个案例的中间结果以支持断点续传"""
        intermediate_dir = output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        case_file = intermediate_dir / f"{case_name}.pkl"
        intermediate_data = {
            "chunks": chunks,
            "edges": list(edges),  # Convert set to list for serialization
            "case_name": case_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "chunk_length": self.chunk_length,
        }

        with open(case_file, "wb") as f:
            pickle.dump(intermediate_data, f)

        print(f"Saved intermediate result for {case_name}: {len(chunks)} chunks")

    def load_intermediate_results(self, output_dir: Path) -> Tuple[Dict, Set, Set]:
        """加载已保存的中间结果"""
        intermediate_dir = output_dir / "intermediate"
        if not intermediate_dir.exists():
            return {}, set(), set()

        all_chunks = {}
        all_edges = set()
        processed_cases = set()

        print("Loading intermediate results...")
        for case_file in intermediate_dir.glob("*.pkl"):
            try:
                with open(case_file, "rb") as f:
                    data = pickle.load(f)
                    all_chunks.update(data["chunks"])
                    all_edges.update(set(data["edges"]))  # Convert back to set
                    processed_cases.add(data["case_name"])
                    print(
                        f"Loaded intermediate result: {data['case_name']} ({len(data['chunks'])} chunks)"
                    )
            except Exception as e:
                print(f"Error loading {case_file}: {str(e)}")
                continue

        return all_chunks, all_edges, processed_cases

    def save_checkpoint(
        self, output_dir: Path, metadata: Optional[Dict] = None
    ) -> None:
        """保存检查点状态"""
        checkpoint_file = output_dir / "checkpoint.json"
        checkpoint_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "chunk_length": self.chunk_length,
            "global_metadata": self.global_metadata.to_dict()
            if metadata is None
            else metadata,
            "total_edges": len(self.global_metadata.all_edges),
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        print(f"Saved checkpoint to {checkpoint_file}")

    def load_checkpoint(self, output_dir: Path) -> bool:
        """加载检查点状态"""
        checkpoint_file = output_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return False

        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            # 恢复全局元数据
            metadata_dict = checkpoint_data["global_metadata"]
            self.global_metadata.service2node_id = metadata_dict["service2node_id"]
            self.global_metadata.node_id2service = metadata_dict["node_id2service"]
            self.global_metadata.log_templates = metadata_dict["log_templates"]
            self.global_metadata.metric_names = metadata_dict["metric_names"]
            self.global_metadata.template2id = metadata_dict["template2id"]

            # 重建services和metrics集合
            self.global_metadata.services = set(
                self.global_metadata.service2node_id.keys()
            )
            self.global_metadata.metrics = set(self.global_metadata.metric_names)

            print(f"Loaded checkpoint from {checkpoint_file}")
            print(f"  Services: {len(self.global_metadata.services)}")
            print(f"  Metrics: {len(self.global_metadata.metrics)}")
            print(f"  Log templates: {len(self.global_metadata.log_templates)}")

            return True

        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_file}: {str(e)}")
            return False

    def get_remaining_cases(
        self, data_packs: List[Path], output_dir: Path
    ) -> List[Path]:
        """获取尚未处理的案例列表"""
        _, _, processed_cases = self.load_intermediate_results(output_dir)

        remaining_cases = []
        for data_pack in data_packs:
            if data_pack.name not in processed_cases:
                remaining_cases.append(data_pack)

        if processed_cases:
            print(f"Found {len(processed_cases)} already processed cases")
            print(f"Remaining {len(remaining_cases)} cases to process")

        return remaining_cases

    def merge_intermediate_results(
        self, output_dir: Path, train_ratio: float = 0.7
    ) -> None:
        """合并所有中间结果并生成最终数据集"""
        all_chunks, all_edges, processed_cases = self.load_intermediate_results(
            output_dir
        )

        if not all_chunks:
            print("No intermediate results found to merge")
            return

        print(f"Merging {len(all_chunks)} chunks from {len(processed_cases)} cases")

        self.global_metadata.all_edges.update(all_edges)
        self.save_dataset(all_chunks, output_dir, train_ratio)

        # 清理中间文件（可选）
        # self.cleanup_intermediate_files(output_dir)

    def cleanup_intermediate_files(self, output_dir: Path) -> None:
        """清理中间文件"""
        intermediate_dir = output_dir / "intermediate"
        if intermediate_dir.exists():
            import shutil

            shutil.rmtree(intermediate_dir)
            print("Cleaned up intermediate files")

        checkpoint_file = output_dir / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("Cleaned up checkpoint file")


def process_log_templates_worker(messages: List[str]) -> List[str]:
    try:
        from drain3 import TemplateMiner
        from drain3.file_persistence import FilePersistence
        from drain3.template_miner_config import TemplateMinerConfig
        import logging

        persistence = FilePersistence("data/drain_templates")
        miner_config = TemplateMinerConfig()
        miner_config.load("drain.ini")
        template_miner = TemplateMiner(persistence, config=miner_config)

        processed_messages = []
        for msg in messages:
            line = str(msg).strip()
            if not line:
                continue

            result = template_miner.add_log_message(line)
            template = result.get("template_mined")
            if template is None:
                logging.warning(f"Failed to extract template for: {line}")
                continue
            if template.strip():
                processed_messages.append(template)

        return processed_messages
    except Exception as e:
        logging.error(f"Error processing log templates: {e}")
        return []


def collect_metadata_worker(data_pack_path: Path, chunk_length: int) -> Dict:
    try:
        case_processor = CaseProcessor(chunk_length, GlobalMetadata())
        data_files = case_processor.derive_filenames(data_pack_path)

        services = set()
        metrics = set()
        log_messages = []

        file_types = [
            "normal_log",
            "abnormal_log",
            "normal_metric",
            "abnormal_metric",
            "normal_trace",
            "abnormal_trace",
        ]

        for file_type in file_types:
            if not data_files[file_type].exists():
                continue

            try:
                df = pd.read_parquet(data_files[file_type])

                if "service_name" in df.columns:
                    service_mask = df["service_name"].notna()
                    if service_mask.any():
                        services.update(df.loc[service_mask, "service_name"].unique())

                if "metric" in df.columns:
                    metrics.update(df["metric"].unique())

                del df
            except Exception as e:
                logging.warning(
                    f"Error reading {file_type} from {data_pack_path.name}: {e}"
                )
                continue

        for file_type in ["normal_log", "abnormal_log"]:
            if not data_files[file_type].exists():
                continue

            try:
                df = pd.read_parquet(data_files[file_type])
                if "message" in df.columns:
                    messages = df["message"].astype(str).str.strip()
                    valid_messages = messages[messages != ""].tolist()
                    log_messages.extend(valid_messages)

                del df

            except Exception as e:
                logging.warning(
                    f"Error reading messages from {file_type} in {data_pack_path.name}: {e}"
                )
                continue

        return {
            "case_name": data_pack_path.name,
            "services": services,
            "metrics": metrics,
            "log_messages": log_messages,
        }
    except Exception as e:
        logging.error(f"Error collecting metadata from {data_pack_path.name}: {e}")
        return {
            "case_name": data_pack_path.name,
            "services": set(),
            "metrics": set(),
            "log_messages": [],
        }


def process_case_worker(
    data_pack_path: Path, chunk_length: int, metadata_dict: Dict
) -> Dict:
    try:
        global_metadata = GlobalMetadata()
        global_metadata.service2node_id = metadata_dict["service2node_id"]
        global_metadata.node_id2service = metadata_dict["node_id2service"]
        global_metadata.log_templates = metadata_dict["log_templates"]
        global_metadata.metric_names = metadata_dict["metric_names"]
        global_metadata.template2id = metadata_dict["template2id"]

        processor = CaseProcessor(chunk_length, global_metadata)
        return processor.process_case(data_pack_path)
    except Exception as e:
        logging.error(f"Error in worker processing {data_pack_path.name}: {e}")
        return {"chunks": {}, "edges": set()}


def create_dataset(
    data_root: str,
    cases_file: str,
    output_dir: str,
    max_cases: Optional[int] = None,
    chunk_length: int = 10,
    train_ratio: float = 0.7,
    n_workers: Optional[int] = None,
    progress_manager=None,
    enable_checkpointing: bool = True,
    resume_from_checkpoint: bool = True,
) -> int:
    cases_df = pd.read_parquet(cases_file)
    case_names = (
        cases_df["datapack"].head(max_cases).tolist()
        if max_cases
        else cases_df["datapack"].tolist()
    )
    data_packs = [Path(data_root) / name for name in case_names]

    builder = DatasetBuilder(chunk_length=chunk_length)

    output_path = Path(output_dir)
    if resume_from_checkpoint and enable_checkpointing:
        if builder.load_checkpoint(output_path):
            print("Resuming from checkpoint...")

            all_chunks, all_edges, processed_cases = builder.load_intermediate_results(
                output_path
            )

            if len(processed_cases) == len(data_packs):
                print("All cases already processed! Merging final results...")
                builder.merge_intermediate_results(output_path, train_ratio)
                return len(all_chunks)

            print(
                f"Found {len(processed_cases)} processed cases, continuing from where we left off..."
            )
        else:
            print("No checkpoint found, starting fresh...")

    if progress_manager:
        progress_manager.add_task("metadata", "Building Global Metadata", 100)
        progress_manager.update_task("metadata", status=TaskStatus.RUNNING)

    builder.build_global_metadata(
        data_packs, Path(output_dir), n_workers, enable_checkpointing
    )

    if progress_manager:
        progress_manager.update_task(
            "metadata", status=TaskStatus.COMPLETED, progress=100.0
        )

    if progress_manager:
        progress_manager.add_task("processing", "Processing Cases", len(data_packs))
        progress_manager.update_task("processing", status=TaskStatus.RUNNING)

    all_chunks = builder.process_cases_parallel(
        data_packs, Path(output_dir), n_workers, enable_checkpointing
    )

    if progress_manager:
        progress_manager.update_task(
            "processing", status=TaskStatus.COMPLETED, progress=100.0
        )

    if progress_manager:
        progress_manager.add_task("saving", "Saving Dataset", 100)
        progress_manager.update_task("saving", status=TaskStatus.RUNNING)

    builder.save_dataset(all_chunks, Path(output_dir), train_ratio)

    if progress_manager:
        progress_manager.update_task(
            "saving", status=TaskStatus.COMPLETED, progress=100.0
        )

    return len(all_chunks)
