from pathlib import Path
from typing import Dict, Any, Optional
from dynaconf import Dynaconf  # type: ignore


class Config:
    """简单的配置管理类，基于 dynaconf"""

    def __init__(self, config_file: Optional[str] = None):
        # 默认配置文件路径
        if config_file is None:
            config_file = "settings.toml"

        # 创建 dynaconf 实例
        self._settings = Dynaconf(
            settings_files=[config_file] if Path(config_file).exists() else [],
        )

    def get(self, key: str):
        res = self._settings.get(key, None)  # type: ignore
        assert res is not None, f"Configuration key '{key}' not found"
        return res

    def set(self, key: str, value: Any):
        self._settings.set(key, value)  # type: ignore

    def update(self, updates: Dict[str, Any]):
        """批量更新配置"""
        for key, value in updates.items():
            self.set(key, value)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key in self._settings.keys():  # type: ignore
            result[key] = self._settings.get(key)  # type: ignore
        return result
