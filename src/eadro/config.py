import os
from pathlib import Path
from typing import Dict, Any, Optional
from dynaconf import Dynaconf  # type: ignore


def _get_settings() -> Dynaconf:
    """Initialize and return Dynaconf settings object"""
    config_dir = Path(__file__).parent.parent.parent

    return Dynaconf(
        settings_files=[
            str(config_dir / "settings.toml"),
        ],
        environments=True,
        env_switcher="ENV_FOR_DYNACONF",
        envvar_prefix="EADRO",
        load_dotenv=True,
        merge_enabled=True,
    )


# Global settings instance
settings = _get_settings()


class Config:
    """
    Simple configuration wrapper for backward compatibility.

    This provides a minimal interface that maintains compatibility
    with existing code while using Dynaconf underneath.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Optional path to additional config file
        """
        self._settings = settings

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
