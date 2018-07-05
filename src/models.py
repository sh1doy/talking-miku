# model defining
import chainer
import numpy as np
from chainer import links as L
from chainer import functions as F
from chainer import Chain
from chainer import cuda
from chainer import initializers
from utils import sequence_embed

cuda.get_device(0).use()


class BiLSTM_Encoder(Chain):

    def __init__(self, n_layers, dim_E, dim_rep, dropout=0.2):
        super(BiLSTM_Encoder, self).__init__()
        with self.init_scope():
            self.LSTM = L.NStepBiLSTM(n_layers, dim_E, dim_rep, 0.2)
            self.hh = [L.Linear(dim_rep * 2, dim_rep) for _ in range(n_layers)]
            self.cc = [L.Linear(dim_rep * 2, dim_rep) for _ in range(n_layers)]
            for e, link in enumerate(self.hh):
                self.add_link(str(e) + "h", link)
            for e, link in enumerate(self.cc):
                self.add_link(str(e) + "c", link)
            self.n_layers = n_layers

    def __call__(self, hx, cx, xs):
        hy, cy, ys = self.LSTM(hx, cx, xs)
        hy = F.stack([self.hh[i](
            F.concat([hy[i * 2], hy[i * 2 + 1]], -1)) for i in range(self.n_layers)], 0)
        cy = F.stack([self.cc[i](
            F.concat([cy[i * 2], cy[i * 2 + 1]], -1)) for i in range(self.n_layers)], 0)
        return hy, cy, ys


class Seq2seq(Chain):

    def __init__(self, vocab, SEQ_LEN, dim_E, dim_rep, n_layers):
        super(Seq2seq, self).__init__(
            E=L.EmbedID(vocab + 1, dim_E, initializers.HeNormal(), -1),
            F=L.EmbedID(vocab + 1, dim_E, initializers.HeNormal(), -1),
            encoder=BiLSTM_Encoder(n_layers, dim_E, dim_rep, 0.5),
            decoder=L.NStepLSTM(n_layers, dim_E, dim_rep, 0.5),
            W=L.Linear(dim_rep, vocab)
        )

    def encode(self, seq):
        """
        labels, relations, coefs: array
        texts: list of list
        """

        # Encode
        seq = [s[::-1] for s in seq]
        seq_embedded = sequence_embed(self.E, seq)
        hx, cx, _ = self.encoder(None, None, seq_embedded)
        return(hx, cx)

    def get_loss(self, seq, texts):

        hx, cx = self.encode(seq)

        # Decode
        eos = self.xp.array([0], 'i')
        ys_in = [F.concat([eos, text], axis=0) for text in texts]
        ys_out = [F.concat([text, eos], axis=0) for text in texts]

        texts_embed = sequence_embed(self.F, ys_in)
        batch = len(texts)

        _, _, os = self.decoder(hx, cx, texts_embed)  # cx: (n_layers, batch, dim) <- WTF

        # Loss
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(
            F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch

        return(loss)

    def translate(self, seq, max_length=150):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            batch = len(seq)
            hx, cx = self.encode(seq)
            ys = self.xp.full(batch, 0, 'i')
            h, c = hx, cx
            result = []
            for i in range(max_length):
                eys = self.F(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)
            result = cuda.to_cpu(
                self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)
            # Remove EOS taggs
            outs = []
            for y in result:
                inds = np.argwhere(y == 0)
                if len(inds) > 0:
                    y = y[:inds[0, 0]]
                outs.append(y)
            return(outs)


class Classifier(Chain):

    def __init__(self, vocab, SEQ_LEN, dim_E, dim_rep, n_layers):
        super(Classifier, self).__init__(
            E=L.Linear(vocab + 1, dim_E, initialW=initializers.HeNormal(), nobias=True),
            encoder=L.NStepLSTM(n_layers, dim_E, dim_rep, 0.2),
            W=L.Linear(dim_rep, 1)
        )
        self.diag = self.xp.eye(vocab + 1, dtype="float32")

    def classify(self, seq_onehot):
        seq_embedded = sequence_embed(self.E, seq_onehot)
        sy, _, _ = self.encoder(None, None, seq_embedded)
        y = self.W(sy[0])
        y = F.flatten(y)
        return(y)

    def onehot(self, seq):
        seq_onehot = [self.diag[s.tolist()] for s in seq]
        return seq_onehot

    def get_loss(self, seq, y):
        self.diag = self.xp.array(self.diag, dtype="float32")
        seq_onehot = self.onehot(seq)
        logit = self.classify(seq_onehot)
        loss = F.sigmoid_cross_entropy(logit, y)
        return loss

    def predict(self, seq):
        self.diag = self.xp.array(self.diag, dtype="float32")
        seq_onehot = self.onehot(seq)
        logit = self.classify(seq_onehot)
        logit = F.sigmoid(logit)
        return logit
