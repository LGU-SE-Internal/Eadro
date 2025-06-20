#!/usr/bin/env python3
"""
改进版主运行脚本 - 规范化的Eadro模型训练脚本
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
import torch
import dgl
import typer
from torch.utils.data import Dataset, DataLoader

from src.eadro.utils import load_chunks, read_json, dump_params, dump_scores, seed_everything
from src.eadro.base import BaseModel
from src.eadro.config import Config, load_config_from_args


class ChunkDataset(Dataset):
    """改进的数据集类"""

    def __init__(self, chunks: dict, node_num: int, edges: list):
        self.data = []
        self.idx2id = {}

        for idx, chunk_id in enumerate(chunks.keys()):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]

            # 创建图
            try:
                graph = dgl.graph(edges, num_nodes=node_num)
            except Exception:
                # Fallback for different DGL versions
                graph = dgl.DGLGraph()
                graph.add_nodes(node_num)
                if edges:
                    graph.add_edges(edges[0], edges[1])

            # 添加节点数据
            graph.ndata["logs"] = torch.FloatTensor(chunk["logs"])
            graph.ndata["metrics"] = torch.FloatTensor(chunk["metrics"])
            graph.ndata["traces"] = torch.FloatTensor(chunk["traces"])

            self.data.append((graph, chunk["culprit"]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def get_chunk_id(self, idx: int) -> str:
        """获取chunk ID"""
        return self.idx2id[idx]


def get_device(use_gpu: bool) -> torch.device:
    """获取计算设备"""
    if use_gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")


def collate_fn(batch):
    """数据批处理函数"""
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def setup_logging(log_file: str = None):
    """设置日志"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=handlers,
    )


def load_data(config: Config) -> tuple:
    data_dir = Path(config.get("chunks_dir")) / config.get("data")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(str(metadata_path))
    else:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    config.set("event_num", metadata["event_num"])
    config.set("node_num", metadata["node_num"])
    config.set("metric_num", metadata["metric_num"])
    config.set("chunk_length", metadata["chunk_length"])

    train_chunks, test_chunks = load_chunks(str(data_dir))

    edges = metadata.get("edges", [])

    return train_chunks, test_chunks, edges, metadata


def create_data_loaders(
    train_chunks: dict, test_chunks: dict, node_num: int, edges: list, config: Config
) -> tuple:
    train_dataset = ChunkDataset(train_chunks, node_num, edges)
    test_dataset = ChunkDataset(test_chunks, node_num, edges)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size"),
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size"),
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_model(config: Config, evaluation_epoch: int = 10) -> tuple:
    """训练模型"""
    # 设置随机种子
    seed_everything(config.get("random_seed"))

    # 获取设备
    device = get_device(config.get("gpu"))

    # 加载数据
    train_chunks, test_chunks, edges, metadata = load_data(config)

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        train_chunks, test_chunks, config.get("node_num"), edges, config
    )

    # 创建模型
    model = BaseModel(
        device=device,
        **config.to_dict(),
    )

    # 训练模型
    scores, converge = model.fit(
        train_loader, test_loader, evaluation_epoch=evaluation_epoch
    )

    return scores, converge


app = typer.Typer()


@app.command()
def main(
    # 训练参数
    random_seed: int = typer.Option(42, help="Random seed"),
    gpu: bool = typer.Option(True, help="Use GPU"),
    epochs: int = typer.Option(50, help="Training epochs"),
    batch_size: int = typer.Option(256, help="Batch size"),
    lr: float = typer.Option(0.001, help="Learning rate"),
    patience: int = typer.Option(10, help="Early stopping patience"),
    
    # 融合参数
    self_attn: bool = typer.Option(True, help="Use self attention"),
    fuse_dim: int = typer.Option(128, help="Fusion dimension"),
    alpha: float = typer.Option(0.5, help="Loss combination weight"),
    locate_hiddens: List[int] = typer.Option([64], help="Localization hidden dims"),
    detect_hiddens: List[int] = typer.Option([64], help="Detection hidden dims"),
    
    # 源模型参数
    log_dim: int = typer.Option(16, help="Log embedding dimension"),
    trace_kernel_sizes: List[int] = typer.Option([2], help="Trace conv kernel sizes"),
    trace_hiddens: List[int] = typer.Option([64], help="Trace hidden dimensions"),
    metric_kernel_sizes: List[int] = typer.Option([2], help="Metric conv kernel sizes"),
    metric_hiddens: List[int] = typer.Option([64], help="Metric hidden dimensions"),
    graph_hiddens: List[int] = typer.Option([64], help="Graph hidden dimensions"),
    attn_head: int = typer.Option(4, help="Attention heads for GAT"),
    activation: float = typer.Option(0.2, help="LeakyReLU negative slope"),
    data: str = typer.Option(..., help="Data directory name"),
    result_dir: str = typer.Option("result/", help="Result directory"),
    chunks_dir: str = typer.Option("chunks/", help="Chunks directory"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Config file path"),
):
    
    config_dict = {
        "random_seed": random_seed,
        "gpu": gpu,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "self_attn": self_attn,
        "fuse_dim": fuse_dim,
        "alpha": alpha,
        "locate_hiddens": locate_hiddens,
        "detect_hiddens": detect_hiddens,
        "log_dim": log_dim,
        "trace_kernel_sizes": trace_kernel_sizes,
        "trace_hiddens": trace_hiddens,
        "metric_kernel_sizes": metric_kernel_sizes,
        "metric_hiddens": metric_hiddens,
        "graph_hiddens": graph_hiddens,
        "attn_head": attn_head,
        "activation": activation,
        "data": data,
        "result_dir": result_dir,
        "chunks_dir": chunks_dir,
    }

    # 创建配置
    if config_file and Path(config_file).exists():
        config = Config(config_file)
        # 用命令行参数更新配置文件中的值
        for key, value in config_dict.items():
            config.set(key, value)
    else:
        config = Config()
        for key, value in config_dict.items():
            config.set(key, value)

    # 设置日志
    result_dir_path = Path(config.get("result_dir"))
    result_dir_path.mkdir(parents=True, exist_ok=True)

    hash_id = dump_params(config.to_dict())
    log_file = result_dir_path / hash_id / "running.log"
    setup_logging(str(log_file))

    logging.info(f"Starting training with hash_id: {hash_id}")
    logging.info(f"Configuration: {config.to_dict()}")

    try:
        # 训练模型
        scores, converge = train_model(config)

        # 保存结果
        dump_scores(config.get("result_dir"), hash_id, scores, converge)

        logging.info(f"Training completed successfully. Hash ID: {hash_id}")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    app()
