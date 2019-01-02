from keras.models import load_model
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import _pickle as pickle
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EOS = 0
PAD = 1

with open("in_.pickle", mode='rb') as f:
    in_ = pickle.load(f)
with open("out_.pickle", mode='rb') as f:
    out_ = pickle.load(f)
with open("vocab_dict.pickle", mode='rb') as f:
    vocab_dict = pickle.load(f)
with open("r_vocab_dict.pickle", mode='rb') as f:
    r_vocab_dict = pickle.load(f)

max_in_length = max([len(i) for i in in_])
max_out_length = max([len(i) for i in out_])
max_length = max(max_in_length, max_out_length) + 1

batch_size=len(in_)
vocab_size=len(vocab_dict) - 2
hidden_layer_dim=50
max_length=15

model = load_model('model.h5')

def num(texts):
    new_texts = []
    for count, text in enumerate(texts):
        text = " ".split(text)
        text_len = len(text)
        arr = []
        for i in range(max_length):
            if i< text_len and vocab_dict.get(text[i]) is not None:
                arr.append(vocab_dict[text[i]])
            else:
                arr.append(PAD)
        new_texts.append(arr)
    return new_texts

def reverse_num(texts):
    new_texts = []
    for count, text in enumerate(texts):
        text_len = len(text)
        arr = []
        for i in range(max_length):
            if i< text_len and r_vocab_dict.get(text[i]) is not None:
                arr.append(r_vocab_dict[text[i]])
            else:
                arr.append(" ")
        new_texts.append(" ".join(arr))
    return new_texts

def decode(in_):
    out_ = np.ones((1,max_length)).astype("int16")
    out_[0,0] = EOS
    for i in range(max_length):
        out_seq = model.predict([in_, out_], batch_size=1)
        if np.argmax(out_seq[0,i]) == EOS or i == max_length-1:
            return out_
        else:
            out_[0,i+1] = np.argmax(out_seq[0, i])


while True:
    try:
        input_seq = [input()]
        in_ = np.array(num(input_seq))
        out_seq = decode(in_)
        output = reverse_num(out_seq)
        print(output[0])
    except EOFError:
        break
