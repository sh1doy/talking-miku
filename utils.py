import MeCab
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


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

    def encode(self, seq):
        seq = [token for token in seq]
        return [1] + [self.word2id[word] for word in seq] + [2]

    def decode(self, seq):
        indice = np.where(np.array(seq) == 2)[0][0]
        seq = seq[:indice]
        res = "".join([self.id2word[word] for word in seq if word not in [0, 1, 2]]).replace("▁", "")
        return res

    def decode_batch(self, seqs):
        res = [self.decode(seq) for seq in list(seqs)]
        return res


def get_batch(x, y, seq_len):
    x = pad_sequences(x, seq_len, padding="post", truncating="post")
    y_in = pad_sequences([line[:-1] for line in y], seq_len, padding="post", truncating="post")
    y_out = pad_sequences([line[1:] for line in y], seq_len, padding="post", truncating="post")
    return [x, y_in], y_out
