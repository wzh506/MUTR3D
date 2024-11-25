#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29401}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_tracker.py $CONFIG --launcher pytorch ${@:3}
# bash tools/dist_train_tracker.sh plugin/track/configs/resnet101_fpn_3frame.py 8 --work-dir=work_dirs/experiment_name