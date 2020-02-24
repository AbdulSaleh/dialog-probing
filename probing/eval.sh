#!/bin/bash

# Example usage bash probing/eval.sh trecquestion 0
TASK_NAME=$1
CUDA=$2

TASK="parlai.probing_tasks.${TASK_NAME}.agents"

for DATASET in dailydialog wikitext-103
do
    DIRS=`ls trained/${DATASET}`
    for dir in $DIRS
    do
    if [[ $dir != 'old' ]]
    then
        if [[ $dir == *'transformer'* ]]
        then
            m="transformer"
        elif [[ $dir == *'seq2seq_att'* ]]
        then
            m="seq2seq_att"
        elif [[ $dir == *'seq2seq'* ]]
        then
            m="seq2seq"
        fi

        if [[ $dir == *'large'* ]] || [[ $dir == *'finetuned'* ]]
        then
            BATCH=1024
        else
            BATCH=512
        fi

        mf="trained/${DATASET}/${dir}/${m}"
        command="CUDA_VISIBLE_DEVICES=${CUDA} python examples/eval_model.py -t ${TASK} -mf ${mf} --batchsize ${BATCH} --probe all"
        echo $command
        #eval "$command"
    fi
    done
done