#!/bin/bash -ex
export ALGORITHM=${ALGORITHM:-eadro}
export CHECKPOINT_PATH=checkpoints/train_dataset_3/best_model.ckpt
LOGURU_COLORIZE=0 .venv/bin/python main.py container run