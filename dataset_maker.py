import os
import glob
from PIL import Image
import numpy as np
import _pickle as pickle
import random
import matplotlib.pyplot as plt

x = np.arange(-20,20,0.01)
x += np.array([random.random()/100 for i in range(len(x))])
y = np.sinc(x)
plt.plot(x,y)
plt.show()

train_input = []
train_output = []
test_input = []
test_output = []

test_indices = random.sample(list(range(len(x))), int(len(x)/4))
test_input = x[test_indices]
test_output = y[test_indices]
train_input = np.delete(x,list(test_indices))
train_output = np.delete(y, list(test_indices))

with open('train_input.pickle', mode='wb') as f:
    pickle.dump(train_input[:,np.newaxis], f)
with open('train_output.pickle', mode='wb') as f:
    pickle.dump(train_output[:,np.newaxis], f)
with open('test_input.pickle', mode='wb') as f:
    pickle.dump(test_input[:,np.newaxis], f)
with open('test_output.pickle', mode='wb') as f:
    pickle.dump(test_output[:,np.newaxis], f)