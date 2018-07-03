import MeCab
import numpy as np
from chainer import functions as F
from chainer.dataset import to_device


def read_file(file):
    with open(file, "r") as f:
        res = f.readlines()
        res = [line.replace("\n", "") for line in res]
    return res


class Parser:
    def __init__(self):
        self.m = MeCab.Tagger("-Ochasen")

    def parse(self, line):
        parsed = [word.split("\t")[0] for word in self.m.parse(line).split("\n")[:-1]][:-1]
        return parsed


class Tokenizer:
    def __init__(self, word2id, id2word):
        self.word2id = word2id
        self.id2word = id2word

    def get_id(self, word):
        try:
            return self.word2id[word]
        except KeyError as e:
            return 3

    def encode(self, seq):
        seq = [token for token in seq]
        return np.array([1] + [self.get_id(word) for word in seq] + [2], "int32")

    def decode(self, seq):
        res = "".join([self.id2word[word] for word in seq if word not in [0, 1, 2]]).replace("▁", "")
        return res

    def decode_batch(self, seqs):
        res = [self.decode(seq) for seq in list(seqs)]
        return res


def to_device0(x):
    return(to_device(0, x))


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs
