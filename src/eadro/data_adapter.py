from datetime import datetime, timezone, timedelta
import pickle
import json
import os
import random
import shutil
import traceback
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

import numpy as np
import polars as pl
from tqdm import tqdm
from loguru import logger
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from rcabench.openapi import InjectionApi, ApiClient, Configuration

from .utils import CacheManager, timeit


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
            logger.warning(f"Failed to extract template for: {line}")
            return ""
        return template

    def save_cache(self):
        self._cache_manager.save()


class CaseProcessor:
    def __init__(self, chunk_length: int, global_metadata: GlobalMetadata):
        self.chunk_length = chunk_length
        self.global_metadata = global_metadata

    def _process_time_column(self, df: pl.DataFrame) -> pl.DataFrame:
        if "time" not in df.columns:
            return df

        try:
            time_dtype = df.select(pl.col("time")).dtypes[0]
            if str(time_dtype).startswith("Utf8") or str(time_dtype).startswith(
                "String"
            ):
                result = df.with_columns(
                    [pl.col("time").str.to_datetime().dt.offset_by("8h").alias("time")]
                )
            elif (
                "datetime" in str(time_dtype).lower()
                or "timestamp" in str(time_dtype).lower()
            ):
                result = df.with_columns(
                    [pl.col("time").dt.offset_by("8h").alias("time")]
                )
            else:
                result = df.with_columns(
                    [pl.col("time").cast(pl.Datetime).dt.offset_by("8h").alias("time")]
                )

            # Immediate cleanup
            del df
            gc.collect()
            return result
        except Exception as e:
            logger.warning(f"Error processing time column: {e}")
            return df

    def _assign_chunk_indices(
        self, df: pl.DataFrame, intervals: List[Tuple]
    ) -> pl.DataFrame:
        chunk_expr = pl.lit(None, dtype=pl.Int32)
        for chunk_idx, (start_time, end_time) in enumerate(intervals):
            chunk_expr = (
                pl.when((pl.col("time") >= start_time) & (pl.col("time") <= end_time))
                .then(pl.lit(chunk_idx))
                .otherwise(chunk_expr)
            )

        return df.with_columns([chunk_expr.alias("chunk_idx")]).filter(
            pl.col("chunk_idx").is_not_null()
        )

    def derive_filenames(self, data_pack: Path) -> Dict[str, Path]:
        return {
            "abnormal_log": data_pack / "converted" / "abnormal_logs.parquet",
            "normal_log": data_pack / "converted" / "normal_logs.parquet",
            "abnormal_metric": data_pack / "converted" / "abnormal_metrics.parquet",
            "normal_metric": data_pack / "converted" / "normal_metrics.parquet",
            "abnormal_metric_sum": data_pack
            / "converted"
            / "abnormal_metrics_sum.parquet",
            "normal_metric_sum": data_pack / "converted" / "normal_metrics_sum.parquet",
            "abnormal_trace": data_pack / "converted" / "abnormal_traces.parquet",
            "normal_trace": data_pack / "converted" / "normal_traces.parquet",
            "env": data_pack / "converted" / "env.json",
            "injection": data_pack / "converted" / "injection.json",
        }

    def get_time_intervals(
        self, data_files: Dict[str, Path]
    ) -> List[Tuple[datetime, datetime]]:
        with open(data_files["env"]) as f:
            env_data = json.load(f)

        start_time = datetime.fromtimestamp(
            int(env_data["NORMAL_START"]), tz=timezone.utc
        )
        end_time = datetime.fromtimestamp(
            int(env_data["ABNORMAL_END"]), tz=timezone.utc
        )

        intervals = []
        current_time = start_time
        window_duration = timedelta(minutes=2)

        while current_time < end_time:
            window_end = current_time + window_duration
            intervals.append((current_time, min(window_end, end_time)))
            current_time = window_end

        return intervals

    def process_logs(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        node_num = len(self.global_metadata.service2node_id)
        event_num = len(self.global_metadata.log_templates) + 1
        result = np.zeros((len(intervals), node_num, event_num))

        for file_type in ["normal_log", "abnormal_log"]:
            df = self._safe_read_parquet(data_files[file_type], file_type)
            if df is None:
                continue

            try:
                df = self._process_time_column(df)

                # 预处理：过滤有效服务
                filtered_df = df.filter(
                    pl.col("service_name").is_in(
                        list(self.global_metadata.service2node_id.keys())
                    )
                )
                del df  # 立即释放原始df
                gc.collect()

                if filtered_df.height == 0:
                    del filtered_df
                    continue

                # 映射service到node_id和message到template_id
                mapped_df = filtered_df.with_columns(
                    [
                        pl.col("service_name")
                        .replace(self.global_metadata.service2node_id, default=-1)
                        .alias("node_id"),
                        pl.col("message")
                        .cast(pl.Utf8)
                        .str.strip_chars()
                        .alias("message_str"),
                    ]
                ).filter(pl.col("node_id") != -1)
                del filtered_df  # 立即释放过滤后的df
                gc.collect()

                # 映射template_id
                template_mapped_df = mapped_df.with_columns(
                    [
                        pl.col("message_str")
                        .replace(self.global_metadata.template2id, default=0)
                        .alias("template_id")
                    ]
                )
                del mapped_df  # 立即释放映射后的df
                gc.collect()

                # 为每个时间段分配chunk_idx
                chunk_assigned_df = self._assign_chunk_indices(
                    template_mapped_df, intervals
                )
                del template_mapped_df  # 立即释放模板映射后的df
                gc.collect()

                if chunk_assigned_df.height == 0:
                    del chunk_assigned_df
                    continue

                # 使用groupby进行向量化计数
                counts = chunk_assigned_df.group_by(
                    ["chunk_idx", "node_id", "template_id"]
                ).len()
                del chunk_assigned_df  # 立即释放chunk分配后的df
                gc.collect()

                # 批量更新结果矩阵
                for row in counts.iter_rows(named=True):
                    chunk_idx = row["chunk_idx"]
                    node_id = row["node_id"]
                    template_id = row["template_id"]
                    count = row["len"]
                    result[chunk_idx, node_id, template_id] += count

                del counts  # 立即释放计数结果
                gc.collect()

            except Exception as e:
                logger.warning(f"Error processing {file_type}: {e}")
                if "df" in locals():
                    del df
                gc.collect()
                continue

        return result

    def process_metrics(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        node_num = len(self.global_metadata.service2node_id)
        metric_num = len(self.global_metadata.metric_names)
        result = np.zeros((len(intervals), node_num, self.chunk_length, metric_num))

        for file_type in ["normal_metric", "abnormal_metric"]:
            df = self._safe_read_parquet(data_files[file_type], file_type)
            if df is None:
                continue

            try:
                df = self._process_time_column(df)

                service_mapped_df = df.with_columns(
                    [
                        pl.coalesce(
                            [
                                pl.col("attr.k8s.deployment.name"),
                                pl.col("attr.k8s.replicaset.name"),
                            ]
                        ).alias("service_name")
                    ]
                )
                del df  # 立即释放原始df
                gc.collect()

                filtered_df = service_mapped_df.filter(
                    pl.col("service_name").is_in(
                        list(self.global_metadata.service2node_id.keys())
                    )
                    & pl.col("metric").is_in(self.global_metadata.metric_names)
                )
                del service_mapped_df  # 立即释放服务映射后的df
                gc.collect()

                if filtered_df.height == 0:
                    del filtered_df
                    continue

                metric_mapping = {
                    metric: idx
                    for idx, metric in enumerate(self.global_metadata.metric_names)
                }
                mapped_df = filtered_df.with_columns(
                    [
                        pl.col("service_name")
                        .replace(self.global_metadata.service2node_id, default=-1)
                        .alias("node_id"),
                        pl.col("metric")
                        .replace(metric_mapping, default=-1)
                        .alias("metric_id"),
                    ]
                ).filter((pl.col("node_id") != -1) & (pl.col("metric_id") != -1))
                del filtered_df  # 立即释放过滤后的df
                gc.collect()

                chunk_assigned_df = self._assign_chunk_indices(mapped_df, intervals)
                del mapped_df  # 立即释放映射后的df
                gc.collect()

                grouped = chunk_assigned_df.group_by(
                    ["chunk_idx", "node_id", "metric_id"]
                ).agg([pl.col("value").sort_by("time").alias("values")])
                del chunk_assigned_df  # 立即释放chunk分配后的df
                gc.collect()

                for row in grouped.iter_rows(named=True):
                    chunk_idx = row["chunk_idx"]
                    node_id = row["node_id"]
                    metric_id = row["metric_id"]
                    values = np.array(row["values"])

                    values = self._normalize_sequence_length(values, self.chunk_length)
                    result[chunk_idx, node_id, :, metric_id] = values

                del grouped  # 立即释放分组结果
                gc.collect()

            except Exception as e:
                logger.warning(f"Error processing {file_type}: {e}")
                if "df" in locals():
                    del df
                gc.collect()
                continue

        return result

    def process_traces(
        self, data_files: Dict[str, Path], intervals: List[Tuple]
    ) -> np.ndarray:
        node_num = len(self.global_metadata.service2node_id)
        result = np.zeros((len(intervals), node_num, self.chunk_length, 1))

        service_mapping = self.global_metadata.service2node_id
        valid_services = list(service_mapping.keys())

        for file_type in ["normal_trace", "abnormal_trace"]:
            try:
                df = self._safe_read_parquet(data_files[file_type], file_type)
                if df is None:
                    continue

                required_cols = ["time", "service_name", "duration"]
                if not all(col in df.columns for col in required_cols):
                    del df
                    continue

                df = self._process_time_column(df)

                filtered_df = df.filter(pl.col("service_name").is_in(valid_services))
                del df

                if filtered_df.height == 0:
                    del filtered_df
                    continue

                mapped_df = filtered_df.with_columns(
                    [
                        pl.col("service_name")
                        .replace(service_mapping, default=-1)
                        .alias("node_id")
                    ]
                )
                del filtered_df

                valid_mapped_df = mapped_df.filter(pl.col("node_id") != -1)
                del mapped_df

                if valid_mapped_df.height == 0:
                    del valid_mapped_df
                    continue

                chunk_assigned_df = self._assign_chunk_indices(
                    valid_mapped_df, intervals
                )
                del valid_mapped_df

                if chunk_assigned_df.height == 0:
                    del chunk_assigned_df
                    continue

                grouped = chunk_assigned_df.group_by(["chunk_idx", "node_id"]).agg(
                    [pl.col("duration").sort_by("time").alias("durations")]
                )
                del chunk_assigned_df

                for row in grouped.iter_rows(named=True):
                    chunk_idx = int(row["chunk_idx"])
                    node_id = int(row["node_id"])
                    durations = np.array(row["durations"], dtype=np.float64)

                    if len(durations) > 0:
                        durations = self._normalize_sequence_length(
                            durations, self.chunk_length
                        )
                        result[chunk_idx, node_id, :, 0] = durations

                del grouped

            except Exception as e:
                logger.warning(f"Error processing {file_type}: {e}")
                if "df" in locals():
                    del df
                gc.collect()
                continue

        return result

    def extract_service_graph(
        self, data_files: Dict[str, Path]
    ) -> Set[Tuple[int, int]]:
        edges = set()
        span_to_service = {}

        # First pass: build span_to_service mapping
        for file_type in ["normal_trace", "abnormal_trace"]:
            df = self._safe_read_parquet(data_files[file_type], file_type)
            if df is None:
                continue

            try:
                valid_rows = df.select(["span_id", "service_name"]).filter(
                    pl.col("span_id").is_not_null()
                    & pl.col("service_name").is_not_null()
                )
                del df

                if valid_rows.height > 0:
                    for row in valid_rows.iter_rows(named=True):
                        span_to_service[row["span_id"]] = row["service_name"]

                del valid_rows

            except Exception as e:
                logger.warning(f"Error in first pass of {file_type}: {e}")
                if "df" in locals():
                    del df
                gc.collect()
                continue

        # Second pass: extract edges
        for file_type in ["normal_trace", "abnormal_trace"]:
            df = self._safe_read_parquet(data_files[file_type], file_type)
            if df is None:
                continue

            try:
                valid_df = df.select(["parent_span_id", "service_name"]).filter(
                    pl.col("parent_span_id").is_not_null()
                    & pl.col("service_name").is_not_null()
                    & pl.col("parent_span_id").is_in(list(span_to_service.keys()))
                )
                del df

                if valid_df.height == 0:
                    del valid_df
                    continue

                parent_mapped_df = valid_df.with_columns(
                    [
                        pl.col("parent_span_id")
                        .replace(span_to_service, default=None)
                        .alias("parent_service")
                    ]
                ).filter(
                    pl.col("parent_service").is_not_null()
                    & (pl.col("parent_service") != pl.col("service_name"))
                )
                del valid_df

                valid_services = set(self.global_metadata.service2node_id.keys())
                filtered_df = parent_mapped_df.filter(
                    pl.col("parent_service").is_in(list(valid_services))
                    & pl.col("service_name").is_in(list(valid_services))
                )
                del parent_mapped_df

                if filtered_df.height > 0:
                    edge_df = filtered_df.with_columns(
                        [
                            pl.col("parent_service")
                            .replace(self.global_metadata.service2node_id)
                            .alias("parent_id"),
                            pl.col("service_name")
                            .replace(self.global_metadata.service2node_id)
                            .alias("current_id"),
                        ]
                    )
                    del filtered_df

                    for row in edge_df.iter_rows(named=True):
                        edges.add((row["parent_id"], row["current_id"]))

                    del edge_df

            except Exception as e:
                logger.warning(f"Error in second pass of {file_type}: {e}")
                if "df" in locals():
                    del df
                gc.collect()
                continue

        return edges

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

        injection_start = datetime.fromisoformat(
            injection_data["start_time"].replace("Z", "+00:00")
        )
        injection_end = datetime.fromisoformat(
            injection_data["end_time"].replace("Z", "+00:00")
        )

        if injection_start.tzinfo is None:
            injection_start = injection_start.replace(tzinfo=timezone.utc)
        else:
            injection_start = injection_start.astimezone(timezone.utc)

        if injection_end.tzinfo is None:
            injection_end = injection_end.replace(tzinfo=timezone.utc)
        else:
            injection_end = injection_end.astimezone(timezone.utc)

        labels = []
        for start_time, end_time in intervals:
            if injection_start < end_time and injection_end > start_time:
                labels.append(self.global_metadata.service2node_id.get(service, -1))
            else:
                labels.append(-1)

        return labels

    @timeit()
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

    def _safe_read_parquet(
        self, file_path: Path, file_type: str
    ) -> Optional[pl.DataFrame]:
        if not file_path.exists():
            return None

        try:
            df = pl.read_parquet(file_path)
            return df
        except Exception as e:
            logger.warning(f"Error reading {file_type} from {file_path}: {e}")
            return None


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
        enable_checkpointing: bool = True,
    ):
        if enable_checkpointing and self.load_checkpoint(output_dir):
            logger.info("Resumed global metadata from checkpoint")
            return

        logger.info(f"Collecting metadata from {len(data_packs)} cases...")

        all_log_messages = []

        for data_pack in tqdm(data_packs, desc="Collecting metadata"):
            try:
                metadata = collect_metadata_worker(data_pack, self.chunk_length)
                self.global_metadata.update_from_case(metadata)
                all_log_messages.extend(metadata["log_messages"])

                if len(all_log_messages) > 200000:
                    all_log_messages = random.sample(all_log_messages, 200000)

            except Exception as e:
                logger.error(f"Error processing metadata: {e}")

        logger.info(
            f"Extracting log templates from {len(all_log_messages)} messages..."
        )

        processed_messages = []
        batch_size = 1000
        message_batches = [
            all_log_messages[i : i + batch_size]
            for i in range(0, len(all_log_messages), batch_size)
        ]

        for batch in tqdm(message_batches, desc="Processing log templates"):
            try:
                batch_results = process_log_templates_worker(batch)
                processed_messages.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing log template batch: {e}")

        template_counts = Counter(processed_messages)
        top_templates = [template for template, _ in template_counts.most_common(100)]

        self.global_metadata.finalize_mappings(top_templates)

        if enable_checkpointing:
            self.save_checkpoint(output_dir)

        logger.info("Global metadata built:")
        logger.info(f"  Services: {len(self.global_metadata.services)}")
        logger.info(f"  Metrics: {len(self.global_metadata.metrics)}")
        logger.info(f"  Log templates: {len(self.global_metadata.log_templates)}")

        del all_log_messages
        del processed_messages

    def process_cases(
        self,
        data_packs: List[Path],
        output_dir: Path,
        enable_checkpointing: bool = True,
    ) -> Dict:
        all_chunks = {}
        all_edges = set()

        if enable_checkpointing:
            all_chunks, all_edges, _ = self.load_intermediate_results(output_dir)

            remaining_data_packs = self.get_remaining_cases(data_packs, output_dir)

            if len(remaining_data_packs) < len(data_packs):
                logger.info(
                    f"Resume from checkpoint: {len(data_packs) - len(remaining_data_packs)} cases already processed"
                )

            data_packs = remaining_data_packs

        if not data_packs:
            logger.info("All cases already processed!")
            self.global_metadata.all_edges.update(all_edges)
            return all_chunks

        logger.info(f"Processing {len(data_packs)} cases serially...")

        metadata_dict = self.global_metadata.to_dict()

        for data_pack in tqdm(data_packs, desc="Processing cases"):
            try:
                result = process_case_worker(
                    data_pack, self.chunk_length, metadata_dict
                )
                if result is None:
                    logger.warning("Worker returned None result")
                    continue

                all_chunks.update(result["chunks"])
                all_edges.update(result["edges"])

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
                logger.error(f"Error processing case: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")

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

        logger.info(f"Dataset saved to {output_dir}")
        logger.info(
            f"Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}"
        )
        logger.info(
            f"Metadata: {metadata['node_num']} nodes, {metadata['event_num']} events, {metadata['metric_num']} metrics"
        )

    def save_intermediate_result(
        self, case_name: str, chunks: Dict, edges: Set, output_dir: Path
    ) -> None:
        intermediate_dir = output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        case_file = intermediate_dir / f"{case_name}.pkl"
        intermediate_data = {
            "chunks": chunks,
            "edges": list(edges),  # Convert set to list for serialization
            "case_name": case_name,
            "timestamp": datetime.now().isoformat(),
            "chunk_length": self.chunk_length,
        }

        with open(case_file, "wb") as f:
            pickle.dump(intermediate_data, f)

        logger.info(f"Saved intermediate result for {case_name}: {len(chunks)} chunks")

    def load_intermediate_results(self, output_dir: Path) -> Tuple[Dict, Set, Set]:
        intermediate_dir = output_dir / "intermediate"
        if not intermediate_dir.exists():
            return {}, set(), set()

        all_chunks = {}
        all_edges = set()
        processed_cases = set()

        for case_file in intermediate_dir.glob("*.pkl"):
            try:
                with open(case_file, "rb") as f:
                    data = pickle.load(f)
                    all_chunks.update(data["chunks"])
                    all_edges.update(set(data["edges"]))
                    processed_cases.add(data["case_name"])
                    logger.info(
                        f"Loaded intermediate result: {data['case_name']} ({len(data['chunks'])} chunks)"
                    )
            except Exception as e:
                logger.info(f"Error loading {case_file}: {str(e)}")
                continue

        return all_chunks, all_edges, processed_cases

    def save_checkpoint(
        self, output_dir: Path, metadata: Optional[Dict] = None
    ) -> None:
        """保存检查点状态"""
        checkpoint_file = output_dir / "checkpoint.json"
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "chunk_length": self.chunk_length,
            "global_metadata": self.global_metadata.to_dict()
            if metadata is None
            else metadata,
            "total_edges": len(self.global_metadata.all_edges),
        }
        os.makedirs(output_dir, exist_ok=True)

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        logger.info(f"Saved checkpoint to {checkpoint_file}")

    def load_checkpoint(self, output_dir: Path) -> bool:
        checkpoint_file = output_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return False

        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            metadata_dict = checkpoint_data["global_metadata"]
            self.global_metadata.service2node_id = metadata_dict["service2node_id"]
            self.global_metadata.node_id2service = metadata_dict["node_id2service"]
            self.global_metadata.log_templates = metadata_dict["log_templates"]
            self.global_metadata.metric_names = metadata_dict["metric_names"]
            self.global_metadata.template2id = metadata_dict["template2id"]

            self.global_metadata.services = set(
                self.global_metadata.service2node_id.keys()
            )
            self.global_metadata.metrics = set(self.global_metadata.metric_names)

            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            logger.info(f"  Services: {len(self.global_metadata.services)}")
            logger.info(f"  Metrics: {len(self.global_metadata.metrics)}")
            logger.info(f"  Log templates: {len(self.global_metadata.log_templates)}")

            return True

        except Exception as e:
            logger.info(f"Error loading checkpoint {checkpoint_file}: {str(e)}")
            return False

    def get_remaining_cases(
        self, data_packs: List[Path], output_dir: Path
    ) -> List[Path]:
        _, _, processed_cases = self.load_intermediate_results(output_dir)

        remaining_cases = []
        for data_pack in data_packs:
            if data_pack.name not in processed_cases:
                remaining_cases.append(data_pack)

        if processed_cases:
            logger.info(f"Found {len(processed_cases)} already processed cases")
            logger.info(f"Remaining {len(remaining_cases)} cases to process")

        return remaining_cases

    def merge_intermediate_results(
        self, output_dir: Path, train_ratio: float = 0.7
    ) -> None:
        """合并所有中间结果并生成最终数据集"""
        all_chunks, all_edges, processed_cases = self.load_intermediate_results(
            output_dir
        )

        if not all_chunks:
            logger.info("No intermediate results found to merge")
            return

        logger.info(
            f"Merging {len(all_chunks)} chunks from {len(processed_cases)} cases"
        )

        self.global_metadata.all_edges.update(all_edges)
        self.save_dataset(all_chunks, output_dir, train_ratio)

    def cleanup_intermediate_files(self, output_dir: Path) -> None:
        intermediate_dir = output_dir / "intermediate"
        if intermediate_dir.exists():
            shutil.rmtree(intermediate_dir)
            logger.info("Cleaned up intermediate files")

        checkpoint_file = output_dir / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Cleaned up checkpoint file")


def process_log_templates_worker(messages: List[str]) -> List[str]:
    try:
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
                logger.warning(f"Failed to extract template for: {line}")
                continue
            if template.strip():
                processed_messages.append(template)

        return processed_messages
    except Exception as e:
        logger.error(f"Error processing log templates: {e}")
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
            "abnormal_metric_sum",
            "normal_metric_sum",
        ]

        for file_type in file_types:
            file_path = data_files[file_type]
            if not file_path.exists():
                continue

            try:
                df = pl.read_parquet(file_path)

                if "service_name" in df.columns:
                    service_df = df.select("service_name").filter(
                        pl.col("service_name").is_not_null()
                    )
                    service_names = service_df.to_series().unique().to_list()
                    services.update(service_names)
                    del service_df

                if "metric" in df.columns:
                    metric_df = df.select("metric")
                    metric_names = metric_df.to_series().unique().to_list()
                    metrics.update(metric_names)
                    del metric_df

                del df

            except Exception as e:
                logger.warning(
                    f"Error reading {file_type} from {data_pack_path.name}: {e}"
                )
                continue

        for file_type in ["normal_log", "abnormal_log"]:
            file_path = data_files[file_type]
            if not file_path.exists():
                continue

            try:
                df = pl.read_parquet(file_path)
                if "message" in df.columns:
                    message_df = df.select("message").filter(
                        pl.col("message").is_not_null()
                        & (pl.col("message").cast(pl.Utf8).str.strip_chars() != "")
                    )
                    messages = message_df.to_series().to_list()
                    log_messages.extend(messages)
                    del message_df

                del df

            except Exception as e:
                logger.warning(
                    f"Error reading messages from {file_type} in {data_pack_path.name}: {e}"
                )
                continue

        case_name = data_pack_path.name
        return {
            "case_name": case_name,
            "services": services,
            "metrics": metrics,
            "log_messages": log_messages,
        }
    except Exception as e:
        logger.error(f"Error collecting metadata from {data_pack_path.name}: {e}")
        return {
            "case_name": data_pack_path.name,
            "services": set(),
            "metrics": set(),
            "log_messages": [],
        }
    finally:
        gc.collect()


def process_case_worker(
    data_pack_path: Path, chunk_length: int, metadata_dict: Dict
) -> Dict:
    try:
        logger.info(f"Starting processing case: {data_pack_path.name}")

        global_metadata = GlobalMetadata()
        global_metadata.service2node_id = metadata_dict["service2node_id"]
        global_metadata.node_id2service = metadata_dict["node_id2service"]
        global_metadata.log_templates = metadata_dict["log_templates"]
        global_metadata.metric_names = metadata_dict["metric_names"]
        global_metadata.template2id = metadata_dict["template2id"]

        processor = CaseProcessor(chunk_length, global_metadata)
        result = processor.process_case(data_pack_path)

        logger.info(
            f"Completed processing case: {data_pack_path.name}, chunks: {len(result.get('chunks', {}))}"
        )
        return result

    except KeyboardInterrupt:
        logger.warning(f"KeyboardInterrupt in worker processing {data_pack_path.name}")
        raise
    except Exception as e:
        logger.error(f"Error in worker processing {data_pack_path.name}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"chunks": {}, "edges": set()}


def create_dataset(
    data_root: str,
    output_dir: str,
    max_cases: Optional[int] = None,
    chunk_length: int = 10,
    train_ratio: float = 0.7,
    enable_checkpointing: bool = True,
    resume_from_checkpoint: bool = True,
) -> int:
    try:
        logger.info("Starting dataset creation...")

        config = Configuration(host="http://10.10.10.220:32080")
        with ApiClient(configuration=config) as client:
            api = InjectionApi(api_client=client)
            resp = api.api_v1_injections_analysis_with_issues_get()

        assert resp.data is not None, "No cases found in the response"
        case_names = list(
            set([item.injection_name for item in resp.data if item.injection_name])
        )[:max_cases]
        data_packs = [Path(data_root) / name for name in case_names]

        logger.info(f"Found {len(data_packs)} data packs to process")

        builder = DatasetBuilder(chunk_length=chunk_length)

        output_path = Path(output_dir)
        if resume_from_checkpoint and enable_checkpointing:
            if builder.load_checkpoint(output_path):
                logger.info("Resuming from checkpoint...")

                all_chunks, all_edges, processed_cases = (
                    builder.load_intermediate_results(output_path)
                )

                if len(processed_cases) == len(data_packs):
                    logger.info("All cases already processed! Merging final results...")
                    builder.merge_intermediate_results(output_path, train_ratio)
                    return len(all_chunks)

                logger.info(
                    f"Found {len(processed_cases)} processed cases, continuing from where we left off..."
                )
            else:
                logger.info("No checkpoint found, starting fresh...")

        logger.info("Building global metadata...")
        builder.build_global_metadata(
            data_packs, Path(output_dir), enable_checkpointing
        )

        logger.info("Processing cases serially...")
        all_chunks = builder.process_cases(
            data_packs, Path(output_dir), enable_checkpointing
        )

        logger.info("Saving final dataset...")
        builder.save_dataset(all_chunks, Path(output_dir), train_ratio)

        logger.info(
            f"Dataset creation completed successfully. Total chunks: {len(all_chunks)}"
        )
        return len(all_chunks)

    except KeyboardInterrupt:
        logger.warning("Dataset creation interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Fatal error in dataset creation: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
