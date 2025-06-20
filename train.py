#!/usr/bin/env python3
"""
改进版主运行脚本 - 规范化的Eadro模型训练脚本
"""

import argparse
import logging
import os
from pathlib import Path
import torch
import dgl
from torch.utils.data import Dataset, DataLoader

from codes.utils import *
from codes.base import BaseModel
from config import Config, load_config_from_args


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
    """加载数据"""
    data_dir = Path(config.get("chunks_dir")) / config.get("data")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 加载元数据
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(str(metadata_path))
    else:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 更新配置
    config.set("event_num", metadata["event_num"])
    config.set("node_num", metadata["node_num"])
    config.set("metric_num", metadata["metric_num"])
    config.set("chunk_length", metadata["chunk_length"])

    # 加载训练和测试数据
    train_chunks, test_chunks = load_chunks(str(data_dir))

    # 获取图结构
    edges = metadata.get("edges", [])

    return train_chunks, test_chunks, edges, metadata


def create_data_loaders(
    train_chunks: dict, test_chunks: dict, node_num: int, edges: list, config: Config
) -> tuple:
    """创建数据加载器"""
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
        event_num=config.get("event_num"),
        metric_num=config.get("metric_num"),
        node_num=config.get("node_num"),
        device=device,
        **config.to_dict(),
    )

    # 训练模型
    scores, converge = model.fit(
        train_loader, test_loader, evaluation_epoch=evaluation_epoch
    )

    return scores, converge


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Eadro Model Training")

    # 训练参数
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--gpu", default=True, type=lambda x: x.lower() == "true", help="Use GPU"
    )
    parser.add_argument("--epochs", default=50, type=int, help="Training epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument(
        "--patience", default=10, type=int, help="Early stopping patience"
    )

    # 融合参数
    parser.add_argument(
        "--self_attn",
        default=True,
        type=lambda x: x.lower() == "true",
        help="Use self attention",
    )
    parser.add_argument("--fuse_dim", default=128, type=int, help="Fusion dimension")
    parser.add_argument(
        "--alpha", default=0.5, type=float, help="Loss combination weight"
    )
    parser.add_argument(
        "--locate_hiddens",
        default=[64],
        type=int,
        nargs="+",
        help="Localization hidden dims",
    )
    parser.add_argument(
        "--detect_hiddens",
        default=[64],
        type=int,
        nargs="+",
        help="Detection hidden dims",
    )

    # 源模型参数
    parser.add_argument(
        "--log_dim", default=16, type=int, help="Log embedding dimension"
    )
    parser.add_argument(
        "--trace_kernel_sizes",
        default=[2],
        type=int,
        nargs="+",
        help="Trace conv kernel sizes",
    )
    parser.add_argument(
        "--trace_hiddens",
        default=[64],
        type=int,
        nargs="+",
        help="Trace hidden dimensions",
    )
    parser.add_argument(
        "--metric_kernel_sizes",
        default=[2],
        type=int,
        nargs="+",
        help="Metric conv kernel sizes",
    )
    parser.add_argument(
        "--metric_hiddens",
        default=[64],
        type=int,
        nargs="+",
        help="Metric hidden dimensions",
    )
    parser.add_argument(
        "--graph_hiddens",
        default=[64],
        type=int,
        nargs="+",
        help="Graph hidden dimensions",
    )
    parser.add_argument(
        "--attn_head", default=4, type=int, help="Attention heads for GAT"
    )
    parser.add_argument(
        "--activation", default=0.2, type=float, help="LeakyReLU negative slope"
    )

    # 数据参数
    parser.add_argument("--data", type=str, required=True, help="Data directory name")
    parser.add_argument("--result_dir", default="../result/", help="Result directory")
    parser.add_argument("--chunks_dir", default="../chunks/", help="Chunks directory")

    # 配置文件
    parser.add_argument("--config", type=str, help="Config file path")

    args = parser.parse_args()

    # 创建配置
    if args.config and Path(args.config).exists():
        config = Config(args.config)
    else:
        config = load_config_from_args(args)

    # 设置日志
    result_dir = Path(config.get("result_dir"))
    result_dir.mkdir(parents=True, exist_ok=True)

    hash_id = dump_params(config.to_dict())
    log_file = result_dir / hash_id / "running.log"
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
    main()
