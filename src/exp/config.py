import os
from pathlib import Path
from typing import Dict, Any, Optional
from dynaconf import Dynaconf  # type: ignore


class Config:
    def __init__(self, config_file: Path):
        self._settings = Dynaconf(
            settings_files=[
                config_file / "settings.toml",
            ],
            environments=True,
            env_switcher="ENV_FOR_DYNACONF",
            envvar_prefix="EADRO",
            load_dotenv=True,
            merge_enabled=True,
        )

        if config_file and Path(config_file).exists():
            additional_settings = Dynaconf(
                settings_files=[config_file],
                merge_enabled=True,
            )
            for key in additional_settings.keys():  # type: ignore
                self._settings.set(key, additional_settings.get(key))  # type: ignore

    def get(self, key: str):
        res = self._settings.get(key, None)  # type: ignore
        assert res is not None, f"Configuration key '{key}' not found"
        return res

    def set(self, key: str, value: Any):
        self._settings.set(key, value)  # type: ignore

    def update(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            self.set(key, value)
