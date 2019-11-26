#!/bin/bash

TASK_NAME=$1
CUDA=$2
BATCH=$3

task="parlai.probing_tasks.${TASK_NAME}.agents"

dirs=`ls trained/dailydialog`
for dir in $dirs
do
if [ $dir != 'old' ]
then
    if [[ $dir == *'transformer'* ]]
    then
    mf="trained/dailydialog/${dir}/transformer"
    # echo "$mf"

    elif [[ $dir == *'seq2seq_att'* ]]
    then
    mf="trained/dailydialog/${dir}/seq2seq_att"
    # echo "$mf"

    elif [[ $dir == *'seq2seq'* ]]
    then
    mf="trained/dailydialog/${dir}/seq2seq"
    # echo "$mf"
    fi
    command="CUDA_VISIBLE_DEVICES=${CUDA} python examples/eval_model.py -t ${task} -mf ${mf} --batchsize ${BATCH} --probe True"
    eval "$command"
fi
done
