#!/bin/bash

# Example usage bash probing/probe.sh trecquestion 0 50 128 0.001 128 dailydialg
TASK_NAME=$1
CUDA=$2
EPOCHS=$3
BATCHSIZE=$4
LR=$5
HIDDEN=$6
DATASET=$7

task="parlai.probing_tasks.${TASK_NAME}.agents"

dirs=`ls trained/${DATASET}`
for dir in $dirs
do
if [ $dir != 'old' ]
then
   # if [[ $dir == *'transformer'* ]]
   # then
   # m="dailydialog/${dir}"
   # # echo "$mf"
   #
   # elif [[ $dir == *'seq2seq_att'* ]]
   # then
   # m="dailydialog/${dir}"
   # # echo "$mf"
   #
   # elif [[ $dir == *'seq2seq'* ]]
   # then
   # m="dailydialog/${dir}"
   # # echo "$mf"
   # fi
    m="${DATASET}/${dir}"
    command="CUDA_VISIBLE_DEVICES=${CUDA} python probing/probe.py -t ${TASK_NAME} -m ${m} -ep ${EPOCHS} -bs ${BATCHSIZE} -lr ${LR} -hidden ${HIDDEN}"
    eval "$command"
fi
done
