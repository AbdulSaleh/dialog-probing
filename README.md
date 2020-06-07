# Probing Neural Dialog Systems for Conversational Understanding
This code accompanies the paper "Probing Neural Dialog Systems for Conversational Understanding" by Saleh, et al., 2020. 

This repo is mostly built on top of [ParlAI](https://parl.ai/). We add functionality for probing open-domain dialog models (RNNs and Transformers). 
Probing evaluates the quality of internal model representations for conversational skills. 

## Setup

Follow same installation instructions as [ParlAI](https://github.com/facebookresearch/ParlAI/tree/d510bc2e10633d5204e1957a6c98cf30aa1be10d). ParlAI requires Python 3 and PyTorch 1.1 or newer. 
You will also need to install [skorch](https://github.com/skorch-dev/skorch/tree/14f374db158ec7a7f4770a2fa9b02b8016d2d6ff) 0.6 which is required by the probing classifier.  

## Examples

This section takes you through an example of how you would train and probe a dialog model. 



## Code Organization

Most of the code for this study exists within the [**probing**](./probing) directory. 

We also augmented the RNN and Transformer modules in [**agents**](./parlai/agents) with probing functions to extract their internal representations. 
 

Run the code below to train the dialog models. You can add `-sh within` flag to train models on dialogs where the order of utterances is shuffled within conversations. 


### Daily Dialog RNN LSTM, ~20M parameters:
python examples/train_model.py  -t dailydialog -m seq2seq --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 300  -eps 60 -veps 1 -vp 10 -bs 64 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300 -mf trained/dailydialog/default_seq2seq/seq2seq --display-examples True -ltim 30 --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl

### Daily Dialog Seq2Seq + Attention lstm, ~20M parameters:
python examples/train_model.py  -t dailydialog -m seq2seq -att general --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 300  -eps 60 -veps 1 -vp 10  -bs 64 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300 -mf trained/dailydialog/default_seq2seq_att/seq2seq_att --display-examples True --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl

### Daily Dialog Seq2Seq + Attention lstm, ~9M parameters:
python examples/train_model.py -t dailydialog -m transformer/generator -bs 64 --optimizer adam -lr 0.001 --lr-scheduler invsqrt --warmup-updates 4000 -eps 35 -veps 1 --embedding-size 300 --n-heads 3 -tr 300 -mf trained/dailydialog/small_default_transformer/transformer --display-examples True -ltim 30 --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl


## Probing experiments

For all of the following commands, replace \<TASK\> with whatever task name you want to probe for. The task names are listed below.

* trecquestion
* multiwoz
* sgd
* dialoguenli
* wnli
* snips
* scenariosa
* dailydialog_topic


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


## Baselines
 Download GloVe Embeddings by running 
 
```python ParlAI/parlai/zoo/glove_vectors/build.py```

