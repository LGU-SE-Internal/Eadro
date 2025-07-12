#!/bin/bash -ex
export ALGORITHM=${ALGORITHM:-eadro}
LOGURU_COLORIZE=0 .venv/bin/python main.py