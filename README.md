# Dialog Model 
Implementation for the probing experiments in "Probing Neural Dialog Systems for Conversational Understanding," a final project for CS281 at Harvard

This code is inspired by and built off of the ParlAI project

## Prerequisites
This section includes installation of required libraries, and downloading pre-trained models.

### Installation
Install Python packages
```
pip install -r requirements.txt
```

### PyTorch Setup
Follow the instructions [here](https://pytorch.org/get-started/locally/) to download PyTorch version (0.4.0) or by running
```bash
pip3 install torch===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Download GloVe Embeddings
```python ParlAI/parlai/zoo/glove_vectors/build.py```

## Model Training

### Daily Dialog Seq2Seq lstm, ~20M parameters:
CUDA_VISIBLE_DEVICES=1 python examples/train_model.py  -t dailydialog -m seq2seq --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 300  -eps 60 -veps 1 -vp 10 -bs 64 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300 -mf trained/dailydialog/default_seq2seq/seq2seq --display-examples True -ltim 30 --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl

### Daily Dialog Seq2Seq + Attention lstm, ~20M parameters:
CUDA_VISIBLE_DEVICES=0 python examples/train_model.py  -t dailydialog -m seq2seq -att general --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 300  -eps 60 -veps 1 -vp 10  -bs 64 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300 -mf trained/dailydialog/default_seq2seq_att/seq2seq_att --display-examples True --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl

### Daily Dialog Seq2Seq lstm, ~20M parameters, utterances shuffled within:
CUDA_VISIBLE_DEVICES=1 python examples/train_model.py  -t dailydialog -m seq2seq --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 300  -eps 60 -veps 1 -bs 64 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300 -mf trained/dailydialog/default_seq2seq_within/seq2seq --display-examples True -ltim 30 --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl -sh within

### Daily Dialog Seq2Seq + Attention lstm, ~20M trainable, utterances shuffled within:
CUDA_VISIBLE_DEVICES=0 python examples/train_model.py  -t dailydialog -m seq2seq -att general --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 300  -eps 60 -veps 1 -bs 64 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300 -mf trained/dailydialog/default_seq2seq_att_within/seq2seq_att --display-examples True --tensorboard_log True --save-after-valid True --embedding-type glove --validation-metric ppl -sh within -ltim 30

## Probing experiments

For all of the following commands, replace <TASK> with whatever task name you want to probe for. The list of tasks names is below.

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
or 
```python probing/glove.py -t trecquestion wnli multinli```

### Generate embeddings probed from models
```
CUDA_VISIBLE_DEVICES=0 python examples/eval_model.py  -mf trained/twitter/small_default_transformer/transformer.checkpoint -t parlai.probing_tasks.<TASK>.agents --batchsize 128 --probe True
```
This probes all the models in trained/dailydialog/:
```bash probing/eval.sh <TASK> CUDA_DEVICE BATCHSIZE```

example usage: bash probing/eval.sh wnli 1 128

### Evaluate probing tasks

```python probing/probe.py -t <TASK> -m twitter/small_default_transformer -ep 200```
or
```python probing/probe.py -t trecquestion -m GloVe -ep 200```

This evaluates all the embeddings in trained/dailydialog except for trained/dailydialog/old
```bash probing/probe.sh TASK CUDA_DEVICE EPOCHS```

example usage: `bash probing/probe.sh wnli 1 150`
