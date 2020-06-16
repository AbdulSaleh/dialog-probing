#!/bin/bash

# Example usage bash probing/eval.sh trecquestion 0 # decoder
TASK_NAME=$1
CUDA=$2
#PROBE=$3

TASK="probing.tasks.${TASK_NAME}.agents"

for DATASET in dailydialog  # wikitext-103
do
    DIRS=`ls trained/${DATASET}`
    for dir in $DIRS
    do
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
            BATCH=500
        else
            BATCH=1000
        fi

        for MODULE in word_embeddings encoder_state combined
        do
            mf="trained/${DATASET}/${dir}/${m}"
            command="CUDA_VISIBLE_DEVICES=${CUDA} python probing/probe_model.py -t ${TASK} -mf ${mf} --batchsize ${BATCH} --probe ${MODULE}"
            echo $command
            eval "$command"
        done

    done
done