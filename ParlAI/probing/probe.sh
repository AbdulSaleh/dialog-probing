#!/bin/bash

# TASK_NAME=$1
# CUDA=$2
# BATCH=$3
#
# MF=
# TASK=

MODEL_DIRS=`ls ../trained/dailydialog`
for dir in $MODE_DIRS
do
if [ $dir != 'old' ]
then
    if [[ $dir == *'transformer'* ]]
    then
    echo "$dir"

    if [[ $dir == *'seq2seq'* ]]
    then
    echo "$dir"
    fi
fi
done

# CUDA_VISIBLE_DEVICES=$CUDA python examples/eval_model.py -t $TASK -mf $MF --batchsize $BATCH --probe True
