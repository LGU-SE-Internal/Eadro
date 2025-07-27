

```bash
DATASET=ts0-ts-basic-service-pod-failure-94xplz \
INPUT_PATH=/mnt/jfs/rcabench_dataset/${DATASET} \
OUTPUT_PATH=/mnt/jfs/rcabench_dataset/${DATASET} \
CHECKPOINT_PATH=/mnt/jfs/experiment_storage/eadro/checkpoints/exp_rcabench_20250727_172235/best_model.ckpt \
./entrypoint.sh
```