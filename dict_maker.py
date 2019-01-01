import os
import glob
import numpy as np
import _pickle as pickle

with open("in_.pickle", mode='rb') as f:
    in_ = pickle.load(f)
with open("out_.pickle", mode='rb') as f:
    out_ = pickle.load(f)
with open("utters_list.pickle", mode='rb') as f:
    utters_list = pickle.load(f)

all_ = utters_list #all_ contains all the dialogue

vocab = [] #vocab contains all the vocabulary
for dialogue in all_:
    vocab.extend(dialogue.split(" "))

vocab = list(set(vocab)) #remove the duplication
vocab.sort() #sorts the vocab
vocab_length = len(vocab) #number of words in the vocabulary
print(vocab_length)

vocab_dict = dict(zip(vocab, list(range(2, vocab_length+2)))) #key:word element:integer
vocab_dict["<EOS>"] = 0 #repersents "EOS"(End of Statement)
vocab_dict[" "] = 1 #represents "space"
r_vocab_dict = dict(zip(list(range(2, vocab_length+2)), vocab)) #key:integer element:word
r_vocab_dict[0] = "<EOS>" #repersents "EOS"(End of Statement)
r_vocab_dict[1] = " " #represents "space"

# stores the dictionaries in pickle format
with open("vocab_dict.pickle", mode='wb') as f:
    pickle.dump(vocab_dict, f)
with open("r_vocab_dict.pickle", mode='wb') as f:
    pickle.dump(r_vocab_dict, f)