#!/bin/bash

# TASK_NAME=$1
# CUDA=$2
# BATCH=$3
#
# MF=
# TASK=

MODEL_DIRS = `ls ../trained/dailydialog`
for DIR in $MODEL_DIRS
if "DIR" -ne "old"
do
  echo "$dir"
done


# CUDA_VISIBLE_DEVICES=$CUDA python examples/eval_model.py -t $TASK -mf $MF --batchsize $BATCH --probe True
