import json
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = ""):
        self.default_config = {
            "random_seed": 42,
            "gpu": True,
            "epochs": 50,
            "batch_size": 256,
            "lr": 0.001,
            "patience": 10,
            "lr_scheduler": "none",
            "lr_step_size": 50,
            "lr_gamma": 0.1,
            "lr_warmup_epochs": 0,
            "lr_min": 1e-6,
            "self_attn": True,
            "fuse_dim": 128,
            "alpha": 0.1,
            "locate_hiddens": [64],
            "detect_hiddens": [64],
            "log_dim": 16,
            "trace_kernel_sizes": [2],
            "trace_hiddens": [64],
            "metric_kernel_sizes": [2],
            "metric_hiddens": [64],
            "graph_hiddens": [64],
            "attn_head": 4,
            "activation": 0.2,
            "chunk_length": 10,
            "test_ratio": 0.3,
            "threshold": 1,
            "data_root": "/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered",
            "result_dir": "../result/",
            "chunks_dir": "../chunks/",
        }

        self.config = self.default_config.copy()

        if config_path and Path(config_path).exists():
            self.load_config(config_path)

    def load_config(self, config_path: str):
        with open(config_path, "r") as f:
            user_config = json.load(f)

        self.config.update(user_config)

    def save_config(self, config_path: str):
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()

    def update(self, updates: Dict[str, Any]):
        self.config.update(updates)


config = Config()


def load_config_from_args(args) -> Config:
    cfg = Config()

    if hasattr(args, "__dict__"):
        args_dict = vars(args)
        cfg.update(args_dict)

    return cfg
