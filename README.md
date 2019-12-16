# Probing Neural Dialog Systems for Conversational Understanding
Implementation for the probing experiments in "Probing Neural Dialog Systems for Conversational Understanding," a final project for CS281 at Harvard.

This code is mostly built on top of [ParlAI](https://parl.ai/). We add functionality for probing different types of models (RNNs, Transformers). We also add and parse a variety of different datasets to probe for conversational understanding. We also include functionality for training models on shuffled datasets to isolate the effect of the dialog structure on downstream model performance. 


## Prerequisites
This section includes installation of required libraries and files.

### Installation

Follow same installation instructions as [ParlAI](https://github.com/facebookresearch/ParlAI)


### Download GloVe Embeddings
```python ParlAI/parlai/zoo/glove_vectors/build.py```

## Model Training

You can run the code below to retrain our probed models. You can add `-sh within` or `-sh across` to trained models on shuffled dialogs either within conversations or across conversations.

### Daily Dialog Seq2Seq lstm, ~20M parameters:
python examples/train_model.py  -t dailydialog -m seq2seq --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 300  -eps 60 -veps 1 -vp 10 -bs 64 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300 -mf trained/dailydialog/default_seq2seq/seq2seq --display-examples True -ltim 30 --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl

### Daily Dialog Seq2Seq + Attention lstm, ~20M parameters:
python examples/train_model.py  -t dailydialog -m seq2seq -att general --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 300  -eps 60 -veps 1 -vp 10  -bs 64 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300 -mf trained/dailydialog/default_seq2seq_att/seq2seq_att --display-examples True --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl

### Daily Dialog Seq2Seq + Attention lstm, ~9M parameters:
python examples/train_model.py -t dailydialog -m transformer/generator -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 35 -veps 1 --embedding-size 300 --n-heads 3 -tr 300 -mf trained/dailydialog/small_default_transformer/transformer --display-examples True -ltim 30 --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl


## Probing experiments

For all of the following commands, replace \<TASK\> with whatever task name you want to probe for. The task names are listed below.

* trecquestion
* act_dailydialog
* multi_woz
* multinli
* sentiment_dailydialog
* shuffle_across_dailydialog
* snips
* squad
* topic_dailydialog
* ushuffle_dailydialog
* wnli

### Generate GloVe embeddings for probing
```python probing/glove.py -t <TASK>```

or, for generating embeddings for multiple tasks at once:

```python probing/glove.py -t trecquestion wnli multinli```

### Generate embeddings probed from models
```
python examples/eval_model.py  -mf trained/dailydialog/seq2seq/seq2seq.checkpoint -t parlai.probing_tasks.<TASK>.agents --batchsize 128 --probe True
```

This generates state vectors to probe all the models in trained/dailydialog/:
```bash probing/eval.sh <TASK> CUDA_DEVICE BATCHSIZE```

example usage: `bash probing/eval.sh wnli 1 128`

### Evaluate probing tasks

```python probing/probe.py -t <TASK> -m dailydialog/transformer -ep 200```

or, for evaluating using the GloVe embeddings

```python probing/probe.py -t <TASK> -m GloVe -ep 200```

This evaluates all the embeddings for models in in trained/dailydialog:

```bash probing/probe.sh TASK CUDA_DEVICE EPOCHS```

example usage: `bash probing/probe.sh wnli 1 150`
