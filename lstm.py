import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.models import load_model, Model
from keras.layers import LSTM, Dense, Input, Embedding
from keras.layers.wrappers import TimeDistributed
import _pickle as pickle
import numpy as np

PAD = 1
EOS = 0

#import conversation data and dictionaries
with open("in_.pickle", mode='rb') as f:
    in_ = pickle.load(f)
with open("out_.pickle", mode='rb') as f:
    out_ = pickle.load(f)
with open("vocab_dict.pickle", mode='rb') as f:
    vocab_dict = pickle.load(f)
with open("r_vocab_dict.pickle", mode='rb') as f:
    r_vocab_dict = pickle.load(f)


max_in_length = max([len(i.split(" ")) for i in in_])
max_out_length = max([len(i.split(" ")) for i in out_])
max_length = max(max_in_length, max_out_length) + 1

batch_size=len(in_)
vocab_size=len(vocab_dict) - 2
hidden_layer_dim=10
max_length=10

def num(texts):
    new_texts = []
    for count, text in enumerate(texts):
        text = " ".split(text)
        text_len = len(text)
        new_texts.append([vocab_dict[text[i]] if i < text_len else PAD for i in range(max_length)])
    return new_texts


x_input = np.array(num(in_))
y_input = np.array(num(["<EOS>" + " " + i for i in out_]))
y_output_temp = np.array(num([i + " " + "<EOS>" if len(i) < max_length else (i[:max_length-1] + " " + "<EOS>") for i in out_]))

encoder_inputs = Input(shape=(max_length,))
embed_x = Embedding(input_dim=vocab_size, output_dim=hidden_layer_dim, input_length=max_length)(encoder_inputs)
_, h, c = LSTM(hidden_layer_dim, return_state=True)(embed_x)
encoder_states = [h, c]

decoder_inputs = Input(shape=(max_length,))
embed_y = Embedding(input_dim=vocab_size, output_dim=hidden_layer_dim, input_length=max_length)(decoder_inputs)
output = LSTM(hidden_layer_dim, return_sequences=True)(embed_y, initial_state=encoder_states)
dense = Dense(vocab_size, activation='softmax')
decoder_outputs = TimeDistributed(dense)(output)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def generator(mini_batch_size):
    while True:
        indices = np.random.randint(batch_size, size=mini_batch_size)
        x_input_ = x_input[indices,:]
        y_input_ = y_input[indices,:]
        y_output = np.eye(vocab_size,dtype="float16")[y_output_temp[indices,:]]
        yield [np.array(x_input_), np.array(y_input_)], np.array(y_output)

history = model.fit_generator(
    generator=generator(mini_batch_size=1),
    steps_per_epoch=1,
    epochs=500,
    verbose=2)

model.save("model.h5")