#!/bin/bash

TASK_NAME=$1
CUDA=$2
BATCH=$3

task="parlai.probing_tasks.${TASK}.agents"

dirs=`ls ../trained/dailydialog`
for dir in $dirs
do
if [ $dir != 'old' ]
then
    if [[ $dir == *'transformer'* ]]
    then
    mf="trained/dailydialog/${dir}/transformer"
    echo "$mf"

    elif [[ $dir == *'seq2seq_att'* ]]
    then
    mf="trained/dailydialog/${dir}/seq2seq_att"
    echo "$mf"

    elif [[ $dir == *'seq2seq'* ]]
    then
    mf="trained/dailydialog/${dir}/seq2seq"
    echo "$mf"
    fi
    command="CUDA_VISIBLE_DEVICES=${CUDA} python examples/eval_model.py -t ${task} -mf ${mf} --batchsize ${BATCH} --probe True"
    echo "$command"
fi
done

# CUDA_VISIBLE_DEVICES=$CUDA python examples/eval_model.py -t $task -mf $MF --batchsize $BATCH --probe True
#CUDA_VISIBLE_DEVICES=2 python examples/eval_model.py  -mf trained/dailydialog/small_default_transformer/transformer -t parlai.probing_tasks.wnli.agents --batchsize 128 --probe True
