# EADRO Dataset Creation CLI Tools

这个项目提供了两个基于typer的命令行工具来创建数据集。

## 工具说明

### 1. create_dataset.py - 完整数据集创建工具

这是主要的数据集创建工具，支持从大量cases中流式创建完整数据集。

#### 基本用法

```bash
# 使用默认参数创建数据集
python create_dataset.py

# 查看所有可用选项
python create_dataset.py --help

# 自定义参数创建数据集
python create_dataset.py \
    --max-cases 50 \
    --batch-size 3 \
    --chunk-length 10 \
    --train-ratio 0.8 \
    --output-dir my_dataset
```

#### 主要参数

- `--data-root`: 包含case数据的根目录 (默认: `/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/`)
- `--cases-file`: 包含case索引的parquet文件 (默认: `/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet`)
- `--output-dir`: 输出目录 (默认: `dataset_output`)
- `--max-cases`: 最大处理的cases数量 (默认: 10, None表示处理所有)
- `--batch-size`: 批处理大小 (默认: 2)
- `--chunk-length`: 数据块长度 (默认: 10)
- `--train-ratio`: 训练集比例 (默认: 0.7)

### 2. test_local_dataset.py - 本地测试工具

这个工具用于使用本地sdata目录中的测试数据来测试数据集创建功能。

#### 基本用法

```bash
# 使用默认参数进行本地测试
python test_local_dataset.py

# 查看所有可用选项
python test_local_dataset.py --help

# 自定义参数进行测试
python test_local_dataset.py \
    --data-root sdata \
    --output-dir my_test_output \
    --chunk-length 15 \
    --train-ratio 0.8
```

#### 主要参数

- `--data-root`: 本地测试数据目录 (默认: `sdata`)
- `--output-dir`: 输出目录 (默认: `test_dataset_output`)
- `--chunk-length`: 数据块长度 (默认: 10)
- `--train-ratio`: 训练集比例 (默认: 0.7)

### 3. create_dataset.py test-local - 集成测试命令

主工具也提供了一个本地测试子命令：

```bash
# 使用主工具的本地测试命令
python create_dataset.py test-local

# 自定义参数
python create_dataset.py test-local \
    --data-root sdata \
    --output-dir test_output \
    --chunk-length 10
```

## 输出文件

所有工具都会在指定的输出目录中创建以下文件：

- `chunk_train.pkl`: 训练数据chunks
- `chunk_test.pkl`: 测试数据chunks  
- `metadata.json`: 数据集元数据信息

## 功能特性

### 流式处理
- 支持大量cases的流式处理
- 内存友好的批处理机制
- 可配置的批处理大小

### 全局数据集构建
- 从多个cases构建统一的全局映射
- 合并所有cases的日志模板、指标和服务映射
- 生成完整的数据集而不是单独的case输出

### 美观的命令行界面
- 使用rich库提供彩色输出和进度条
- 详细的状态信息和错误处理
- 友好的用户体验

### 内存管理
- 分批处理大量数据
- 及时清理临时数据
- 可配置的处理参数

## 使用建议

1. **开始时先使用本地测试**: 确保功能正常工作
   ```bash
   python test_local_dataset.py
   ```

2. **小规模测试**: 先用较少的cases进行测试
   ```bash
   python create_dataset.py --max-cases 5 --batch-size 1
   ```

3. **生产环境**: 根据内存情况调整批处理大小
   ```bash
   python create_dataset.py --max-cases 100 --batch-size 5
   ```

4. **监控资源**: 大规模处理时监控内存和磁盘使用情况

## 故障排除

- 如果内存不足，减少`--batch-size`参数
- 如果处理速度慢，可以增加`--batch-size`参数  
- 确保输出目录有足够的磁盘空间
- 检查数据目录和cases文件是否存在且有权限访问
