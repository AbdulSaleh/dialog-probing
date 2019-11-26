#!/bin/bash

TASK_NAME=$1
CUDA=$2
EPOCHS=$3

task="parlai.probing_tasks.${TASK_NAME}.agents"

dirs=`ls trained/dailydialog`
for dir in $dirs
do
if [ $dir != 'old' ]
then
    if [[ $dir == *'transformer'* ]]
    then
    m="dailydialog/${dir}"
    # echo "$mf"

    elif [[ $dir == *'seq2seq_att'* ]]
    then
    m="dailydialog/${dir}"
    # echo "$mf"

    elif [[ $dir == *'seq2seq'* ]]
    then
    m="dailydialog/${dir}"
    # echo "$mf"
    fi
    command="CUDA_VISIBLE_DEVICES=${CUDA} python probing/probe.py -t ${TASK_NAME} -m ${m} -ep ${EPOCHS}"
    eval "$command"
fi
done
