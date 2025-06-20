# Eadro - 多源数据故障检测与定位模型

## 项目简介

Eadro 是一个基于多源数据（日志、指标、链路追踪）的故障检测和定位模型，使用图神经网络和时序卷积网络进行特征提取和融合。该项目现已适配新的 RCABench 数据格式。

## 项目结构

```
Eadro/
├── codes/                      # 核心代码
│   ├── model.py               # 模型定义
│   ├── base.py                # 基础训练类
│   ├── main.py                # 原始训练脚本
│   ├── utils.py               # 工具函数
│   └── preprocess/            # 数据预处理
│       ├── align.py           # 数据对齐
│       ├── single_process.py  # 单独数据处理
│       └── util.py            # 预处理工具
├── data_adapter.py            # 新数据格式适配器
├── train.py                   # 改进的训练脚本
├── config.py                  # 配置管理
├── 模型输入输出及数据适配计划.md  # 详细文档
└── README.md                  # 项目说明
```

## 模型架构

### 1. 核心组件
- **MultiSourceEncoder**: 多源数据编码器
  - **TraceModel**: 链路追踪数据处理（1D卷积+自注意力）
  - **MetricModel**: 指标数据处理（1D卷积+自注意力）
  - **LogModel**: 日志数据处理（线性嵌入）
  - **GraphModel**: 图神经网络（GAT+全局池化）

- **MainModel**: 主模型
  - **Detecter**: 故障检测器（二分类）
  - **Localizer**: 故障定位器（多分类）

### 2. 输入格式
```python
# 图节点数据
graph.ndata["logs"]     # [batch_size*node_num, event_num]
graph.ndata["metrics"]  # [batch_size*node_num, chunk_length, metric_num]
graph.ndata["traces"]   # [batch_size*node_num, chunk_length, 1]

# 标签
fault_indexs           # [batch_size] - 故障节点索引
```

### 3. 输出格式
```python
{
    "loss": 总损失,
    "y_pred": 预测故障节点列表,
    "y_prob": 真实标签概率矩阵,
    "pred_prob": 预测概率矩阵
}
```

## 新数据适配

### 1. 数据格式
新的 RCABench 数据包含以下格式：

- **指标数据**: `container.memory.major_page_faults`, `service_name`, `time` 等
- **日志数据**: `message`, `service_name`, `trace_id`, `time` 等  
- **链路数据**: `duration`, `service_name`, `span_name`, `time` 等

### 2. 适配流程
1. **服务映射**: 建立 `service_name` 到节点ID的映射
2. **时间窗口**: 将数据按时间切分为固定长度窗口
3. **特征提取**:
   - 日志: 提取模板并使用Hawkes过程建模
   - 指标: 构建多维时序特征矩阵
   - 链路: 提取延迟信息构建时序数据
4. **图构建**: 基于服务调用关系构建图结构

## 使用方法

### 1. 数据预处理
```bash
python data_adapter.py
```

### 2. 模型训练
```bash
# 使用改进的训练脚本
python train.py --data your_data_name --gpu true --epochs 50

# 或使用原始脚本
python codes/main.py --data your_data_name
```

### 3. 配置管理
```python
# 创建配置
from config import Config
config = Config("config.json")

# 修改配置
config.set("batch_size", 128)
config.set("lr", 0.001)
```

## 主要改进

### 1. 代码规范化
- 修复了原代码中的错误（属性名、类型注解等）
- 改进导入结构和代码风格
- 添加类型提示和文档字符串

### 2. 数据适配
- 实现了完整的数据适配流程
- 支持新的parquet数据格式
- 自动构建服务映射和图结构

### 3. 配置管理
- 统一的配置管理系统
- 支持配置文件和命令行参数
- 参数验证和默认值设置

### 4. 模块化设计
- 清晰的模块分离
- 可复用的组件设计
- 易于扩展和维护

## 依赖项

```
torch>=1.8.0
dgl>=0.6.0
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```

## 注意事项

1. **保持模型结构不变**: 本次适配严格保持了原模型的网络结构和训练逻辑
2. **数据格式兼容**: 确保预处理后的数据格式与模型期望输入一致
3. **性能考虑**: 大规模数据处理时注意内存使用
4. **参数设置**: 根据具体数据调整超参数

## 实验结果

模型在新数据上的性能指标：
- HR@1: 故障定位Top-1准确率
- HR@3: 故障定位Top-3准确率  
- HR@5: 故障定位Top-5准确率
- F1: 故障检测F1分数
- NDCG: 归一化折损累积增益

## 扩展开发

### 1. 新数据源
可以通过修改 `data_adapter.py` 支持新的数据源和格式。

### 2. 新模型组件
可以在 `model.py` 中添加新的神经网络组件。

### 3. 新预处理方法
可以在 `preprocess/` 目录下添加新的数据预处理方法。

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
