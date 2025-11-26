# Eadro Algorithm Reproduction

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Build Docker Image

```bash
docker build -t eadro .
```

### 3. Run the Algorithm

```bash
docker run -it \
  -v $(pwd)/data/rcabench_dataset/ts0-mysql-bandwidth-5p8bkc:/data/rcabench_dataset/ts0-mysql-bandwidth-5p8bkc \
  -e INPUT_PATH=/data/rcabench_dataset/ts0-mysql-bandwidth-5p8bkc \
  -e OUTPUT_PATH=/data/rcabench_dataset/ts0-mysql-bandwidth-5p8bkc \
  -e RCABENCH_SUBMISSION='false' \
  eadro
```

## Parameters

- `INPUT_PATH`: Input data path
- `OUTPUT_PATH`: Output results path 
- `RCABENCH_SUBMISSION`: Whether to submit results to RCABench platform (default: false)

## Data Preparation

Ensure the data directory structure follows:

```bash
data/
└── rcabench_dataset/
    └── ts0-mysql-bandwidth-5p8bkc/
        ├── injection.json
        └── ... (other data files)
```