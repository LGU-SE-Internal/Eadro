"""
配置文件 - Eadro模型配置管理
"""

import json
from pathlib import Path
from typing import Dict, Any


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        # 默认配置
        self.default_config = {
            # 训练参数
            "random_seed": 42,
            "gpu": True,
            "epochs": 50,
            "batch_size": 256,
            "lr": 0.001,
            "patience": 10,
            # 融合参数
            "self_attn": True,
            "fuse_dim": 128,
            "alpha": 0.5,
            "locate_hiddens": [64],
            "detect_hiddens": [64],
            # 源模型参数
            "log_dim": 16,
            "trace_kernel_sizes": [2],
            "trace_hiddens": [64],
            "metric_kernel_sizes": [2],
            "metric_hiddens": [64],
            "graph_hiddens": [64],
            "attn_head": 4,
            "activation": 0.2,
            # 数据参数
            "chunk_length": 10,
            "test_ratio": 0.3,
            "threshold": 1,
            # 路径配置
            "data_root": "/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered",
            "result_dir": "../result/",
            "chunks_dir": "../chunks/",
        }

        self.config = self.default_config.copy()

        if config_path and Path(config_path).exists():
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """从文件加载配置"""
        with open(config_path, "r") as f:
            user_config = json.load(f)

        # 更新配置
        self.config.update(user_config)

    def save_config(self, config_path: str):
        """保存配置到文件"""
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def get(self, key: str, default=None):
        """获取配置项"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """设置配置项"""
        self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()

    def update(self, updates: Dict[str, Any]):
        """批量更新配置"""
        self.config.update(updates)


# 创建全局配置实例
config = Config()


def load_config_from_args(args) -> Config:
    """从命令行参数创建配置"""
    cfg = Config()

    # 将argparse的Namespace转换为字典
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
        cfg.update(args_dict)

    return cfg
