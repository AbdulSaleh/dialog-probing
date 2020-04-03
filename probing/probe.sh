#!/bin/bash

# Example usage bash probing/probe.sh trecquestion 0 50 128 0.001 128 10
TASK_NAME=$1
CUDA=$2
EPOCHS=$3
BATCHSIZE=$4
LR=$5
HIDDEN=$6
RUNS=$7

task="parlai.probing_tasks.${TASK_NAME}.agents"

for DATASET in dailydialog #wikitext-103
do
    dirs=`ls trained/${DATASET}`
    for dir in $dirs
    do
        if [ $dir != 'old' ]
        then
            if [[ $dir == *'seq2seq_att'* ]]
            then
                modules=("hierarchical_encoder_state" "hierarchical_encoder_embeddings_state")
            elif [[ $dir == *'transformer'* ]]
            then
                modules=("encoder_embeddings" "hierarchical_encoder_state" "hierarchical_encoder_embeddings_state")
            else
                continue
            fi

            for MODULE in "${arr[@]}"
            do
                m="${DATASET}/${dir}"
                command="CUDA_VISIBLE_DEVICES=${CUDA} python probing/probe.py -t ${TASK_NAME} -p ${MODULE} -m ${m} -ep ${EPOCHS} -r ${RUNS} -bs ${BATCHSIZE} -lr ${LR} -hidden ${HIDDEN}"
                eval "$command"
            done


#           for MODULE in encoder_embeddings encoder_state encoder_embeddings_state
#           do
#                m="${DATASET}/${dir}"
#                command="CUDA_VISIBLE_DEVICES=${CUDA} python probing/probe.py -t ${TASK_NAME} -p ${MODULE} -m ${m} -ep ${EPOCHS} -r ${RUNS} -bs ${BATCHSIZE} -lr ${LR} -hidden ${HIDDEN}"
#                eval "$command"
##                echo "$command"
#           done
         fi
    done
done

#command="CUDA_VISIBLE_DEVICES=${CUDA} python probing/probe.py -t ${TASK_NAME} -m GloVe -ep ${EPOCHS} -r ${RUNS} -bs ${BATCHSIZE} -lr ${LR} -hidden ${HIDDEN}"
#eval "$command"
#echo "$command"