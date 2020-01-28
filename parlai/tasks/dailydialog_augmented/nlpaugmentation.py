# -*- coding: utf-8 -*-
"""NLPAugmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Wluper/dsag/blob/master/NLPAugmentation.ipynb

# Text Augmentation
"""

import os
from tqdm import tqdm

# conversation_dataset = [
#   ["hi","how are you","fine thanks","great. Take care"],
#   ["heya","ugh go away","why are you so mean to me","im just in a bad mood"],
#   ["what is your name","im Carla","nice to meet you Carla","likewise"]
# ]

"""## Shifting Conversations

Varying the Position of Sentences within a conversation by shifting the text to the left / right is a useful trick for sequence data
"""

def shift_examples(examples):
  return [
    example[idx:] for example in examples for idx in range(
      len(example)
    ) if len(example[idx:]) > 1
  ]

"""## Combining Conversations"""

def combine_examples(examples):
  return [
    first_example + second_example for idx,first_example in enumerate(
        examples
    ) for jdx, second_example in enumerate(
        examples
    ) if idx != jdx
  ]

"""## Substituting Synonyms (using Wordnet)"""

import nltk 
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet as wn
from string import punctuation

def synonyms(word, pos_tag):
  return list(
    {
      lemma.replace("_"," ").replace("-"," ") for synset in wn.synsets(
          _clean_word(word),
          pos_tag,
      ) for lemma in synset.lemma_names()
    }
  )

def _clean_word(word):
  return word.lower().strip(punctuation)

def _tokenise(sentence):
  return nltk.word_tokenize(sentence)

def _infer_pos_tags(tokens):
  return [
    (
      token,
      _convert_nltk_to_wordnet_tag(nltk_tag)
    ) for token,nltk_tag in nltk.pos_tag(tokens)
  ]

def _convert_nltk_to_wordnet_tag(pos_tag):
  NOUN = "NN"
  VERB = "VB"
  ADJECTIVE = "JJ"
  ADVERB = "RB"
  if pos_tag.startswith(NOUN):
    return "n"
  if pos_tag.startswith(VERB):
    return "v" 
  if pos_tag.startswith(ADVERB):
    return "r"
  if pos_tag.startswith(ADJECTIVE):
    return "a"

def synonymous_examples(examples, include_verbs = False):
  synonymous = []
  for example in examples:
    for idx, sentence in tqdm(enumerate(example)):
      tokens = _tokenise(sentence)
      tagged_words = _infer_pos_tags(tokens)
      for jdx,word_pos in enumerate(tagged_words):
        word, pos_tag = word_pos
        if pos_tag and (include_verbs or pos_tag != "v"):
          for synonym in synonyms(word, pos_tag):
              new_tokens = tokens[:jdx] + [synonym] + tokens[jdx+1:]
              new_sentence = ' '.join(new_tokens)
              new_example = example[:idx] + [new_sentence] + example[idx+1:]
              synonymous.append(new_example)
  return synonymous

synonymous_examples(
  [
    ["This is a little test", "yes it is"],
    ["this is another test", "no it isn't"]
  ]
)

"""## Back Translating (aka Spinning)"""

from textblob import TextBlob

def _spin_text(text, foreign_language): 
  try: 
    spun_text = _clean_word(
      TextBlob(
        TextBlob(text).translate(
          from_lang="en",
          to=foreign_language
        ).raw
      ).translate(
        from_lang=foreign_language,
        to="en"
      ).raw
    )
    return spun_text if spun_text != _clean_word(text) else None
  except:
    return None

def rephrase_examples(examples):
  rephrased_examples = []
  repeat_rephrasings = []
  for example in tqdm(examples):
    for idx,sentence in enumerate(example):
      sentence_spun_from_spanish = _spin_text(sentence, "es")
      if sentence_spun_from_spanish and sentence_spun_from_spanish not in repeat_rephrasings:
        repeat_rephrasings.append(sentence_spun_from_spanish)
        rephrased_examples.append(
          example[:idx] + [sentence_spun_from_spanish] + example[idx+1:]
        )
      sentence_spun_from_arabic = _spin_text(sentence, "ar")
      if sentence_spun_from_arabic and sentence_spun_from_arabic not in repeat_rephrasings:
        repeat_rephrasings.append(sentence_spun_from_arabic)
        rephrased_examples.append(
          example[:idx] + [sentence_spun_from_arabic] + example[idx+1:]
        )
  return examples + rephrased_examples

"""## Inserting words (using BERT)"""

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

model_name = 'bert-base-uncased'
bert_tokeniser = BertTokenizer.from_pretrained(model_name)
bert_model = BertForMaskedLM.from_pretrained(model_name)

def _format_model_input(text, tokeniser, insert_mask_at_idx):
  tokens = tokeniser.tokenize(
    f"[CLS] {text} [SEP]"
  )
  tokens_with_mask = tokens[:insert_mask_at_idx] + [
    "[MASK]"
  ] + tokens[insert_mask_at_idx:]
  return torch.tensor(
    [
      tokeniser.convert_tokens_to_ids(tokens_with_mask)
    ]
  )

def _format_model_output(model_output, token_idxs, tokeniser, masked_idx):
  tokens = tokeniser.convert_ids_to_tokens(
    token_idxs.tolist()[0]
  )
  tokens[masked_idx] = tokeniser.convert_ids_to_tokens(
    [
      torch.argmax(
        model_output[0, masked_idx]
      ).item()
    ]
  )[0]
  return ' '.join(tokens[1:-1]).replace("##","")

def _insert_mask_and_predict(sentence, model, tokeniser, masked_idx):
  tokens_with_mask_inserted = _format_model_input(
    text = sentence,
    tokeniser = tokeniser,
    insert_mask_at_idx = masked_idx,
  )
  segment_ids = torch.tensor(
    [[0]*len(tokens_with_mask_inserted)]
  )
  with torch.no_grad():
    return _format_model_output(
      model_output = model(
        tokens_with_mask_inserted,
        segment_ids
      ),
      tokeniser = tokeniser,
      token_idxs = tokens_with_mask_inserted,
      masked_idx = masked_idx,
    )

def _insert_words(example):
  new_examples = [example]
  idx = 1
  try:
    while True:
      new_examples.append(
        _insert_mask_and_predict(
          sentence = example,
          model = bert_model,
          tokeniser = bert_tokeniser,
          masked_idx = idx
        )
      )
      idx += 1
  except:
    new_examples.pop()
    return new_examples

def bert_inserted_examples(examples):
  new_examples = []
  for example in tqdm(examples):
    for idx,sentence in enumerate(example):
      for inserted_sentence in _insert_words(sentence):
        new_examples.append(
          example[:idx] + [inserted_sentence] + example[idx+1:]
        )
  return examples + new_examples

"""## Generating longer conversations (using GPT-2)"""

from nltk.tokenize import sent_tokenize
import gpt_2_simple as gpt2
# TODO THis was originall 744M but that didn't fit on my gpu
model_name = "345M"
if not os.path.exists(os.path.join('models', model_name)):
    gpt2.download_gpt2(model_name=model_name)
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(
  sess, 
  model_name=model_name
)

def extend_conversations(examples):
  new_examples = []
  for example in tqdm(examples):
    new_examples.append(_extend_conversation('. '.join(example)))
  return examples + new_examples

def _extend_conversation(conversation_as_string):
  generated_samples = gpt2.generate(
    sess,
    model_name=model_name,
    prefix=conversation_as_string,
    length=100,
    return_as_list = True
  )
  n = len(
    sent_tokenize(
      conversation_as_string
    )
  )
  return sent_tokenize(
    generated_samples[0]
  )[:n+1]


"""# Pipeline"""

def augment_dataset(dataset):
  print('extending')
  dataset = extend_conversations(dataset)
  print('bert inserting')
  dataset = bert_inserted_examples(dataset)
  print('rephrasing')
  dataset = rephrase_examples(dataset)
  print('synonomizing')
  dataset = synonymous_examples(dataset)
  print('sfhiting')
  dataset = shift_examples(dataset)
  return dataset
