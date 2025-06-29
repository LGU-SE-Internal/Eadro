from pathlib import Path
from typing import List

from loguru import logger
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from .utils import CacheManager
import os


class DrainProcessor:
    def __init__(self, conf: str, save_path: str, cache_dir: str = "./cache/drain"):
        os.makedirs(cache_dir, exist_ok=True)
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
