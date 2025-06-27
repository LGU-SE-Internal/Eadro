import math
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch import nn

from dgl.nn.pytorch.conv import GATv2Conv
from dgl.nn.pytorch.glob import GlobalAttentionPooling


class GraphModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        device: str,
        config: Any,
    ) -> None:
        super(GraphModel, self).__init__()
        """
        Graph neural network model using GAT layers and global attention pooling
        
        Args:
            in_dim: Feature dimension of each node
            device: Device to run on
            config: Configuration object
        """
        graph_hiddens = config.get("model.graph.hiddens")
        attn_head = config.get("model.attn_head")
        activation = config.get("model.activation")
        attn_drop = config.get("model.graph.attn_drop")

        layers = []

        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i - 1] if i > 0 else in_dim
            dropout = attn_drop
            layers.append(
                GATv2Conv(
                    in_feats,
                    out_feats=hidden,
                    num_heads=attn_head,
                    attn_drop=dropout,
                    negative_slope=activation,
                    allow_zero_in_degree=True,
                )
            )
            self.maxpool = nn.MaxPool1d(attn_head)

        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))

    def forward(self, graph: Any, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            graph: DGL graph object
            x: Node feature tensor [batch_size*node_num, feature_in_dim]

        Returns:
            Graph-level representation [batch_size, out_dim]
        """
        out = None
        for layer in self.net:
            if out is None:
                out = x
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        return self.pooling(graph, out)  # [bz*node, out_dim] --> [bz, out_dim]


class Chomp1d(nn.Module):
    """Module for cropping excess padding after convolution"""

    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class ConvNet(nn.Module):
    """1D convolutional network for processing time series data"""

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_sizes: List[int],
        dilation: int = 2,
        dev: str = "cpu",
        dropout: float = 0.0,
    ) -> None:
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation**i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                Chomp1d(padding),
            ]
            # Add dropout after each layer except the last one
            if dropout > 0.0 and i < len(kernel_sizes) - 1:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        self.out_dim = num_channels[-1]
        self.network.to(dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, T, in_dim]

        Returns:
            Output tensor [batch_size, T, out_dim]
        """
        x = x.permute(0, 2, 1).float()  # [batch_size, in_dim, T]
        out = self.network(x)  # [batch_size, out_dim, T]
        out = out.permute(0, 2, 1)  # [batch_size, T, out_dim]
        return out


class SelfAttention(nn.Module):
    """Self-attention mechanism module"""

    def __init__(self, input_size: int, seq_len: int) -> None:
        """
        Args:
            input_size: Input size (hidden_size * num_directions)
            seq_len: Sequence length (window_size)
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, window_size, input_size]

        Returns:
            Weighted sum tensor
        """
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (
            torch.bmm(input_tensor, self.atten_w) + self.atten_bias
        )  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor: Optional[torch.Tensor]) -> None:
        """Glorot initialization"""
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)


class TraceModel(nn.Module):
    """Model for processing trace data"""

    def __init__(
        self,
        device: str,
        config: Any,
    ) -> None:
        super(TraceModel, self).__init__()

        trace_hiddens = config.get("model.trace.hiddens")
        trace_kernel_sizes = config.get("model.trace.kernel_sizes")
        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(
            2, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, dev=device
        )

        trace_self_attn = config.get("model.trace.self_attn")
        self.self_attn = trace_self_attn
        if trace_self_attn:
            chunk_length = config.get("model.trace.chunk_length")
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trace data tensor [batch_size, T, 1]

        Returns:
            Processed features [batch_size, out_dim]
        """
        hidden_states = self.net(x)
        if self.self_attn:
            return self.attn_layer(hidden_states)
        return hidden_states[:, -1, :]  # [bz, out_dim]


class MetricModel(nn.Module):
    def __init__(
        self,
        metric_num: int,
        device: str,
        config: Any,
    ) -> None:
        super(MetricModel, self).__init__()
        self.metric_num = metric_num

        metric_hiddens = config.get("model.metric.hiddens")
        metric_kernel_sizes = config.get("model.metric.kernel_sizes")
        metric_dropout = config.get("model.metric.dropout")
        self.out_dim = metric_hiddens[-1]
        in_dim = metric_num

        assert len(metric_hiddens) == len(metric_kernel_sizes)
        self.net = ConvNet(
            num_inputs=in_dim,
            num_channels=metric_hiddens,
            kernel_sizes=metric_kernel_sizes,
            dev=device,
            dropout=metric_dropout,
        )

        metric_self_attn = config.get("model.metric.self_attn")
        self.self_attn = metric_self_attn
        if metric_self_attn:
            chunk_length = config.get("model.metric.chunk_length")
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Metric data tensor [batch_size, T, metric_num]

        Returns:
            Processed features [batch_size, out_dim]
        """
        assert x.shape[-1] == self.metric_num
        hidden_states = self.net(x)
        if self.self_attn:
            return self.attn_layer(hidden_states)
        return hidden_states[:, -1, :]  # [bz, out_dim]


class LogModel(nn.Module):
    """Model for processing log data"""

    def __init__(self, event_num: int, out_dim: int) -> None:
        super(LogModel, self).__init__()
        self.embedder = nn.Linear(event_num, out_dim)

    def forward(self, paras: torch.Tensor) -> torch.Tensor:
        """
        Args:
            paras: Event parameter tensor [batch_size, event_num]

        Returns:
            Embedded features [batch_size, out_dim]
        """
        return self.embedder(paras)


class MultiSourceEncoder(nn.Module):
    """Multi-source data encoder that fuses trace, log, and metric data"""

    def __init__(
        self,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: str,
        config: Any,
    ) -> None:
        super(MultiSourceEncoder, self).__init__()
        self.node_num = node_num
        self.alpha = config.get("model.alpha")

        self.trace_model = TraceModel(
            device=device,
            config=config,
        )
        trace_dim = self.trace_model.out_dim
        log_dim = config.get("model.log_dim")
        self.log_model = LogModel(event_num, log_dim)
        self.metric_model = MetricModel(
            metric_num=metric_num,
            device=device,
            config=config,
        )
        metric_dim = self.metric_model.out_dim
        fuse_in = trace_dim + log_dim + metric_dim

        fuse_dim = config.get("model.fuse_dim")
        if not fuse_dim % 2 == 0:
            fuse_dim += 1
        self.fuse = nn.Linear(fuse_in, fuse_dim)

        self.activate = nn.GLU()
        self.feat_in_dim = int(fuse_dim // 2)

        self.status_model = GraphModel(
            in_dim=self.feat_in_dim,
            device=device,
            config=config,
        )
        self.feat_out_dim = self.status_model.out_dim

    def forward(self, graph: Any) -> torch.Tensor:
        """
        Args:
            graph: DGL graph object containing trace, log, and metric data

        Returns:
            Fused graph-level representation [batch_size, feat_out_dim]
        """
        trace_embedding = self.trace_model(
            graph.ndata["traces"]
        )  # [bz*node_num, trace_dim]
        log_embedding = self.log_model(graph.ndata["logs"])  # [bz*node_num, log_dim]
        metric_embedding = self.metric_model(
            graph.ndata["metrics"]
        )  # [bz*node_num, metric_dim]

        # Fuse multi-source features
        feature = self.activate(
            self.fuse(
                torch.cat((trace_embedding, log_embedding, metric_embedding), dim=-1)
            )
        )  # [bz*node_num, feat_in_dim]
        embeddings = self.status_model(graph, feature)  # [bz, feat_out_dim]
        return embeddings


class FullyConnected(nn.Module):
    """Fully connected network"""

    def __init__(self, in_dim: int, out_dim: int, linear_sizes: List[int]) -> None:
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i - 1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, in_dim]

        Returns:
            Output tensor [batch_size, out_dim]
        """
        return self.net(x)


class MainModel(nn.Module):
    """Main model for anomaly detection and fault localization"""

    def __init__(
        self,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: str,
        config: Any,
    ) -> None:
        super(MainModel, self).__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = config.get("model.alpha")

        self.encoder = MultiSourceEncoder(
            event_num=event_num,
            metric_num=metric_num,
            node_num=node_num,
            device=device,
            config=config,
        )

        # Anomaly detector (binary classification: normal/anomaly)
        detect_hiddens = config.get("model.detection.hiddens")
        self.detector = FullyConnected(self.encoder.feat_out_dim, 2, detect_hiddens).to(
            device
        )
        self.decoder_criterion = nn.CrossEntropyLoss()

        # Fault localizer (multi-classification: which node is faulty)
        locate_hiddens = config.get("model.localization.hiddens")
        self.locator = FullyConnected(
            self.encoder.feat_out_dim, node_num, locate_hiddens
        ).to(device)
        self.locator_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.get_prob = nn.Softmax(dim=-1)

    def forward(self, graph: Any, ground_truth: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass

        Args:
            graph: DGL graph object
            fault_indexs: Fault node indices [batch_size]

        Returns:
            Dictionary containing loss, predictions, and probabilities
        """
        batch_size = graph.batch_size
        embeddings = self.encoder(graph)

        # Construct ground truth labels
        y_prob = torch.zeros((batch_size, self.node_num)).to(self.device)
        for i in range(batch_size):
            if ground_truth[i] > -1:
                y_prob[i, ground_truth[i]] = 1
        y_anomaly = torch.zeros(batch_size).long().to(self.device)
        for i in range(batch_size):
            y_anomaly[i] = int(ground_truth[i] > -1)

        # Fault localization
        locate_logits = self.locator(embeddings)
        locate_loss = self.locator_criterion(
            locate_logits, ground_truth.to(self.device)
        )

        # Anomaly detection
        detect_logits = self.detector(embeddings)
        detect_loss = self.decoder_criterion(detect_logits, y_anomaly)

        # Total loss
        loss = self.alpha * detect_loss + (1 - self.alpha) * locate_loss

        node_probs = self.get_prob(locate_logits.detach()).cpu().numpy()
        y_pred = self.inference(batch_size, node_probs, detect_logits)

        return {
            "loss": loss,
            "y_pred": y_pred,
            "y_prob": y_prob.detach().cpu().numpy(),
            "pred_prob": node_probs,
        }

    def inference(
        self,
        batch_size: int,
        node_probs: np.ndarray,
        detect_logits: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """
        Inference phase for generating predictions

        Args:
            batch_size: Batch size
            node_probs: Node probabilities [batch_size, node_num]
            detect_logits: Detection logits [batch_size, 2]

        Returns:
            List of predicted fault nodes for each sample
        """
        node_list = np.flip(node_probs.argsort(axis=1), axis=1)

        y_pred = []
        for i in range(batch_size):
            if detect_logits is not None:
                detect_pred = (
                    detect_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
                )
                if detect_pred[i] < 1:
                    y_pred.append([-1])  # Predicted as normal
                else:
                    y_pred.append(
                        node_list[i]
                    )  # Predicted as anomaly, return sorted nodes
            else:
                y_pred.append(node_list[i])

        return y_pred
