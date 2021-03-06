"""General utils for probing
"""

import re

import numpy as np
import contractions
from tqdm import tqdm


def remove_contractions(lst):
    """Takes a list of strings and removes contractions
    """
    return [contractions.fix(s) for s in lst]


def re_tokenize(s):
    """Input string. Return list of tokenized words
    """
    return re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;]", s)


def load_glove(path):
    print("Loading GloVe Model")
    with open(path, 'r', encoding='utf8') as f:
        embs = {}
        for line in tqdm(f, total=2.2*(10**6)):
            splitLine = line.split(' ')
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            embs[word] = embedding

    print("Done.", len(embs), " words loaded!")
    return embs


def encode_sent(s, glove, dict):
    s = contractions.fix(s)
    words = re_tokenize(s)

    emb = np.zeros(glove['hi'].shape)
    for w in words:
        if dict is not None and w not in dict:
            # word not in provided dictionary
            continue

        try:
            emb += glove[w]
        except KeyError:
            # word not found in glove
            continue
    lens = 1 if len(words) == 0 else len(words)
    return emb / lens


def encode_glove(sents, glove, dict=None):
    emb_size = len(glove['hi'])
    embs = np.zeros((len(sents), emb_size))
    for i, s in enumerate(sents):
        embs[i] = encode_sent(s, glove, dict)

    return embs
