#!/bin/bash

# Example usage bash probing/eval.sh trecquestion 0 1024 wikitext-103
TASK_NAME=$1
CUDA=$2
BATCH=$3
DATASET=$4

task="parlai.probing_tasks.${TASK_NAME}.agents"

dirs=`ls trained/${DATASET}`
for dir in $dirs
do
if [ $dir != 'old' ]
then
    if [[ $dir == *'transformer'* ]]
    then
    m="transformer"
    # echo "$mf"

    elif [[ $dir == *'seq2seq_att'* ]]
    then
    m="seq2seq_att"
    # echo "$mf"

    elif [[ $dir == *'seq2seq'* ]]
    then
    m="seq2seq"
    # echo "$mf"
    fi
    mf="trained/${DATASET}/${dir}/${m}"
    command="CUDA_VISIBLE_DEVICES=${CUDA} python examples/eval_model.py -t ${task} -mf ${mf} --batchsize ${BATCH} --probe True"
    eval "$command"
fi
done
