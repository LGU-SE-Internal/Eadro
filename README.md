

```bash
DATASET=ts0-ts-basic-service-pod-failure-94xplz \
INPUT_PATH=/mnt/jfs/rcabench_dataset/${DATASET} \
OUTPUT_PATH=/mnt/jfs/rcabench_dataset/${DATASET} \
CHECKPOINT_PATH=/mnt/jfs/experiment_storage/eadro/checkpoints/exp_rcabench_20250727_172235/best_model.ckpt \
./entrypoint.sh
```

```sh
ln -s /mnt/jfs/rcabench_dataset/ ./data/
export RCABENCH_BASE_URL=http://10.10.10.220:32080
export RCABENCH_USERNAME=admin
export RCABENCH_PASSWORD=admin123
export CHECKPOINT_PATH=checkpoints/train_dataset_17/best_model.ckpt
sudo -E .venv/bin/python client.py train-by-id --dataset-id 17

sudo -E .venv/bin/python run.py batch-test --label 8.26run
```