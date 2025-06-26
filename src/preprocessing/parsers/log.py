from pathlib import Path
from typing import List

from loguru import logger
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from ..utils import CacheManager


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
        results = []
        for sentence in sentences:
            line = str(sentence).strip()
            if not line:
                results.append("")
                continue

            cached_result = self._cache_manager.get(line)
            if cached_result is not None:
                results.append(cached_result)
                continue

            template = self._extract_template(line)
            self._cache_manager.set(line, template)
            results.append(template)

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
