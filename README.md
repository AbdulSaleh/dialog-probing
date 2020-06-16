# Probing Neural Dialog Systems for Conversational Understanding
This code accompanies the paper "Probing Neural Dialog Systems for Conversational Understanding" by Saleh, et al., 2020. 

This repo is built on top of [ParlAI](https://parl.ai/). We add functionality for probing open-domain dialog models (RNNs and Transformers). 
Probing evaluates the quality of internal model representations for conversational skills. 

## Setup

Follow same installation instructions as [ParlAI](https://github.com/facebookresearch/ParlAI/tree/d510bc2e10633d5204e1957a6c98cf30aa1be10d). ParlAI requires Python 3 and PyTorch 1.1 or newer. After cloning this repo, remember to run 

```python setup.py develop```

You will also need to install [skorch](https://github.com/skorch-dev/skorch/tree/14f374db158ec7a7f4770a2fa9b02b8016d2d6ff) 0.6 which is required by the probing classifier.  

## Example Usage

This section takes you through an example of how you would train and probe a dialog model. 

1. You will first need a model to probe. Let's train a small RNN on the DailyDialog dataset:

    ```
    python examples/train_model.py -t dailydialog -m seq2seq --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 128 -eps 60 -veps 1 -vp 10 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300  -mf trained/dailydialog/seq2seq/seq2seq --display-examples True -ltim 30 --tensorboard_log True --validation-metric ppl
    ```

    To train on perturbed (i.e. shuffled) data, add the flag ``-sh within``. See ParlAI's [documentation](http://parl.ai.s3-website.us-east-2.amazonaws.com/docs/index.html) for more information about training dialog models. 

2. You will then generate and save the vector representations to be used as features by the probing classifier. 
Let's generate and save the ``encoder_state`` vectors for the TREC question classification task:

    ```
    python probing/probe_model.py -mf trained/dailydialog/seq2seq/seq2seq -t probing.tasks.trecquestion.agents --probe encoder_state 
    ```
    This will automatically download the required task data and save the generated representations at ``trained/dailydialog/seq2seq/probing/encoder_state/trecquestion``.

3. Now you can run the probing classifier to evaluate the quality of the generated representations by running:

    ```
    python probing/eval_probing.py -m trained/dailydialog/seq2seq -t trecquestion --probing-module encoder_state --max_epochs 50 --runs 30
    ```
    This trains the probing classifier (an MLP) on the generated representations. The final results are saved at ``trained/dailydialog/seq2seq/probing/encoder_state/trecquestion/results.json``. 

### Baselines

1. You might also want to generate the GloVe word embedding baselines. You can do this by running:

    ```
    python probing/glove.py -t trecquestion
    ```
    This will automatically download the GloVe embeddings and save the generated representations to ``trained/GloVe/probing/trecquestion``

2. Now you need to run the probing classifier on these generated representations using:

    ```
    python probing/eval_probing.py -m GloVe -t trecquestion --max_epochs 50 --runs 30
    ```
    The final results are saved at ``trained/GloVe/probing/trecquestion/results.json``. 

## Probing tasks
The supported probing tasks are:

* trecquestion 
* multiwoz
* sgd
* dialoguenli
* wnli
* snips
* scenariosa
* dailydialog_topic

### Adding new probing tasks
New probing tasks need to be in the following format:
```
text: <utterance1> \n
<utterance2> \t episode_done:True \n  
text: <utterance1> \n
<utterance2> \n
<utterance3> \t episode_done:True \n 
... 
```
New probing tasks need to be added to [**probing/tasks**](./probing/tasks) 
and [glove.py](./probing/glove.py).

The code is best suited for tasks where the label is based on:
* all utterances in a dialog (like DailyDialog Topic)
* or the interaction between two utterances (like DialogueNLI)
* or the last utterance in a dialog (like ScenarioSA)

See section [3.2]() in the paper for more info. 


## Code Organization

Most of the code for this study exists within the [**probing**](./probing) directory. 

We also augmented [torch_generator_agent.py](./parlai/core/torch_generator_agent.py) with probing functions that extract a model's internal representations. 


## Reference

If you use this code please cite our paper:
```
@article{saleh2020probing,
    author= {Saleh, Abdelrhman and Deutsch, Tovly and Casper, Stephen and Belinkov, Yonatan and Shieber, Stuart},
    title= {Probing Neural Dialog Models for Conversational Understanding},
    journal={Second Workshop on NLP for Conversational AI},
    year={2020}
    }
```

