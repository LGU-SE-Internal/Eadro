from pathlib import Path
from typing import List
from loguru import logger
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from .utils import CacheManager
from .base import DatasetMetadata
import polars as pl
from datetime import datetime
import numpy as np


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

        cached_result = self._cache_manager.get(line)
        if cached_result is not None:
            return cached_result

        template = self._extract_template(line)
        self._cache_manager.set(line, template)

        self.save_cache()
        return template

    def process_batch(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        results = []
        cache_hits = 0
        new_templates = {}

        # 预处理：去重和清理
        unique_sentences = {}
        for i, sentence in enumerate(sentences):
            line = str(sentence).strip()
            if not line:
                results.append("")
                continue

            if line not in unique_sentences:
                unique_sentences[line] = []
            unique_sentences[line].append(i)

        # 初始化结果数组
        results = [""] * len(sentences)

        # 批量查找缓存
        for line, indices in unique_sentences.items():
            cached_result = self._cache_manager.get(line)
            if cached_result is not None:
                cache_hits += len(indices)
                for idx in indices:
                    results[idx] = cached_result
            else:
                # 处理新的模板
                template = self._extract_template(line)
                new_templates[line] = template
                for idx in indices:
                    results[idx] = template

        # 批量更新缓存
        if new_templates:
            for line, template in new_templates.items():
                self._cache_manager.set(line, template)

        if new_templates:
            self.save_cache()

        return results

    def _extract_template(self, line: str) -> str:
        result = self._template_miner.add_log_message(line)
        template = result.get("template_mined")
        if template is None:
            logger.warning(f"Failed to extract template for: {line}")
            return ""
        return template

    def save_cache(self):
        self._cache_manager.save()


def preaggregate_logs(
    df: pl.DataFrame,
    start_time: datetime,
    end_time: datetime,
    metadata: DatasetMetadata,
    drain_config_path: str,
    drain_save_path: str,
) -> np.ndarray:
    """Pre-aggregate log data at second-level granularity for the entire time range"""
    total_seconds = int((end_time - start_time).total_seconds())
    num_services = len(metadata.services)
    num_templates = len(metadata.log_templates) + 1  # +1 for unseen templates

    if df.height == 0:
        return np.zeros((num_services, total_seconds, num_templates))

    # Create a DrainProcessor instance for this worker
    drain = DrainProcessor(conf=drain_config_path, save_path=drain_save_path)

    # Batch process messages to get templates
    messages = df["message"].to_list()
    templates = drain.process_batch(messages)

    # Add templates and time buckets
    df_with_templates = df.with_columns(pl.Series("template", templates))
    df_with_buckets = df_with_templates.with_columns(
        [
            ((pl.col("time") - start_time).dt.total_seconds())
            .floor()
            .cast(pl.Int32)
            .alias("time_bucket")
        ]
    ).filter((pl.col("time_bucket") >= 0) & (pl.col("time_bucket") < total_seconds))

    if df_with_buckets.height == 0:
        return np.zeros((num_services, total_seconds, num_templates))

    result = np.zeros((num_services, total_seconds, num_templates))

    # Calculate counts by service, template, and time bucket
    counts = df_with_buckets.group_by(["service_name", "template", "time_bucket"]).agg(
        pl.count().alias("count")
    )

    service_lookup = metadata.service_name_to_id
    template_lookup = metadata.log_template_to_id

    for row in counts.iter_rows(named=True):
        service_name = row["service_name"]
        template = row["template"]
        time_bucket = row["time_bucket"]
        count = row["count"]

        service_id = service_lookup.get(service_name)
        if service_id is not None and 0 <= time_bucket < total_seconds:
            template_id = template_lookup.get(
                template, 0
            )  # Default to 0 for unseen templates
            if 0 <= template_id < num_templates:
                result[service_id, time_bucket, template_id] += count

    return result
