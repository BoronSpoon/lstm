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

#import conversation data and dictionaries
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

#counts the max number of vocabulary in a dialogue for in_ and out_
max_in_length = max([len(i.split(" ")) for i in in_])
max_out_length = max([len(i.split(" ")) for i in out_])
max_length = max(max_in_length, max_out_length) + 1 #this is the max number of vocabulary for both in_ and out_ plus 1 for spare

batch_size=len(in_) #represents the number of dialogue pairs
vocab_size=len(vocab_dict) - 2 #number of vocabulary except for PAD and EOS
hidden_layer_dim=10 #number of hidden_layers
max_length=10 #max words for the input sequence

#turns dialogue into list of keys(integers)
def num(texts):
    new_texts = []
    for count, text in enumerate(texts):
        text = " ".split(text)
        text_len = len(text)
        new_texts.append([vocab_dict[text[i]] if i < text_len else PAD for i in range(max_length)]) #fills the list with PAD until the last element
    return new_texts


x_input = np.array(num(in_)) #input of encoder
y_input = np.array(num(["<EOS>" + " " + i for i in out_])) #input of decoder (starts with EOS(End of Sequence))
y_output_temp = np.array(num([i + " " + "<EOS>" if len(i) < max_length else (i[:max_length-1] + " " + "<EOS>") for i in out_])) #output of decoder (ends with EOS(End of Sequence))

def generator(mini_batch_size):
    while True:
        indices = np.random.randint(batch_size, size=mini_batch_size)
        x_input_ = x_input[indices,:]
        y_input_ = y_input[indices,:]
        y_output = np.eye(vocab_size)[y_output_temp[indices,:]]
        yield [np.array(x_input_), np.array(y_input_)], np.array(y_output)

loss = []

#resets training periodically to prevent outofmemory error
for i in range(1000):
    print("batch", str(i))
    model = load_model('model.h5')

    early_stopping = keras.callbacks.EarlyStopping(patience=0, verbose=1)

    #training phase
    history = model.fit_generator(
        generator=generator(mini_batch_size=10),
        steps_per_epoch=1,
        epochs=500,
        verbose=2)

    #saves the model
    model.save("model.h5")
    model.save('model.h5')

    #loss.append(mean(history.history['loss']))
    #plt.plot(loss)
    #plt.pause(.001)

