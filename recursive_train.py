import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.models import load_model, Model
from keras.layers import LSTM, Dense, Input, Embedding
from keras.layers.wrappers import TimeDistributed
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

with open("in_.pickle", mode='rb') as f:
    in_ = pickle.load(f)
with open("out_.pickle", mode='rb') as f:
    out_ = pickle.load(f)
with open("vocab_dict.pickle", mode='rb') as f:
    vocab_dict = pickle.load(f)
with open("r_vocab_dict.pickle", mode='rb') as f:
    r_vocab_dict = pickle.load(f)

PAD = 1
EOS = 0

max_in_length = max([len(i) for i in in_])
max_out_length = max([len(i) for i in out_])
max_length = max(max_in_length, max_out_length) + 1

batch_size=69541
vocab_size=2363
hidden_layer_dim=10
max_length=30


def num(texts):
    new_texts = []
    for count, text in enumerate(texts):
        text_len = len(text)
        new_texts.append([vocab_dict[text[i]] if i < text_len else PAD for i in range(max_length)])
    return new_texts


x_input = np.array(num(in_))
y_input = np.array(num(["§" + i for i in out_]))
y_output_temp = np.array(num([i + "§" if len(i) < max_length else (i[:max_length-1] + "§") for i in out_]))

def generator(mini_batch_size):
    while True:
        indices = np.random.randint(batch_size, size=mini_batch_size)
        x_input_ = x_input[indices,:]
        y_input_ = y_input[indices,:]
        y_output = np.eye(vocab_size)[y_output_temp[indices,:]]
        yield [np.array(x_input_), np.array(y_input_)], np.array(y_output)

loss = []

for i in range(1000):
    print("batch", str(i))
    model = load_model('model.h5')

    early_stopping = keras.callbacks.EarlyStopping(patience=0, verbose=1)

    history = model.fit_generator(
        generator=generator(mini_batch_size=10),
        steps_per_epoch=1,
        epochs=1000,
        verbose=2)

    model.save('model.h5')

    #loss.append(mean(history.history['loss']))
    #plt.plot(loss)
    #plt.pause(.001)

