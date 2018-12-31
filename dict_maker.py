import os
import glob
import numpy as np
import _pickle as pickle

with open("in_.pickle", mode='rb') as f:
    in_ = pickle.load(f)
with open("out_.pickle", mode='rb') as f:
    out_ = pickle.load(f)

all = in_ + [out_[-1]]

vocab = set([])
for i in all:
    c = set(i)
    vocab = vocab | c
vocab.remove(" ")

vocab = list(vocab)

#2363
vocab_dict = dict(zip(vocab, list(range(1, 2362+1))))
vocab_dict["§"] = 0
vocab_dict[" "] = 1
r_vocab_dict = dict(zip(list(range(1, 2362+1)), vocab))
r_vocab_dict[0] = "§"
r_vocab_dict[1] = " "

with open("vocab_dict.pickle", mode='wb') as f:
    pickle.dump(vocab_dict, f)
with open("r_vocab_dict.pickle", mode='wb') as f:
    pickle.dump(r_vocab_dict, f)