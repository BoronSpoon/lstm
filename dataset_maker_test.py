import codecs
import re
import _pickle as pickle

in_ = []
out_ = []
dic = {}

def line_to_dic():
with open(path) as f:
    lines = [s.strip() for s in f.readlines()]

def conv_pair_maker(i):


for i in range(1,129+1):
    utter = conv_pair_maker(i)
    in_.extend(utter[:-1])
    out_.extend(utter[1:])

with open("in_.pickle", mode='wb') as f:
    pickle.dump(in_, f)
with open("out_.pickle", mode='wb') as f:
    pickle.dump(out_, f)

print(len(in_), len(out_))