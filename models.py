import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
keras = tf.keras


class Seq2seq:
    def __init__(self):
        encoder_inputs = Input([None], dtype="int32", name="x")
        E_embed = Embedding(VOCAB, EMBEDDING_SIZE, mask_zero=True, name="E_embed")(encoder_inputs)
        encoder1 = LSTM(NUM_UNITS, return_state=True, return_sequences=True, dropout=.2, recurrent_dropout=.2)
        encoder2 = LSTM(NUM_UNITS, return_state=True, dropout=.2, recurrent_dropout=.2)
        out, *mid_states1 = encoder1(E_embed)
        out, *mid_states2 = encoder2(out)
        # End2end learning
        decoder_inputs = Input(shape=[None], dtype="int32", name="y_")
        F_embed = Embedding(VOCAB, EMBEDDING_SIZE, mask_zero=True, name="F_embed")(decoder_inputs)
        decoder1 = LSTM(NUM_UNITS, return_sequences=True, return_state=True, dropout=.2, recurrent_dropout=.2)
        decoder2 = LSTM(NUM_UNITS, return_sequences=True, return_state=True, dropout=.2, recurrent_dropout=.2)
        decoder_outputs, *decoder_states1 = decoder1(F_embed, initial_state=mid_states1)
        decoder_outputs, *decoder_states2 = decoder2(decoder_outputs, initial_state=mid_states2)
        decoder_dense = Dense(VOCAB, activation='softmax', name="output_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        self.training_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        # Single Encoder
        self.encoder_model = Model(inputs=encoder_inputs, outputs=mid_states1 + mid_states2)
        # Single Decoder
        decoder_states = [Input([NUM_UNITS]) for _ in range(4)]
        d_out, *new_decoder_states1 = decoder1(F_embed, initial_state=decoder_states[0:2])
        d_out, *new_decoder_states2 = decoder2(d_out, initial_state=decoder_states[2:4])
        new_decoder_outputs = decoder_dense(d_out)

        self.decoder_model = Model(inputs=[decoder_inputs] + decoder_states,
                                   outputs=[new_decoder_outputs] + new_decoder_states1 + new_decoder_states2)

        self.training_model.compile(Adam(1e-3), loss='sparse_categorical_crossentropy')
#         self.encoder_model.compile(Adam(1e-4), loss='sparse_categorical_crossentropy')
#         self.decoder_model.compile(Adam(1e-4), loss='sparse_categorical_crossentropy')

    # generate target given source sequence
    def predict_sequence(self, source, n_steps, mode="greedy", alpha=1.0):
        # encode
        state = self.encoder_model.predict(source)
        # start of sequence input
        x = np.array([[1] for _ in range(len(source))])
        # collect predictions
        output = list()
        for t in range(n_steps):
            # predict next char
            x, *state = self.decoder_model.predict([x] + state)
            if mode == "greedy":
                x = x.argmax(-1)
            elif mode == "random":
                next_x = []
                for i in range(len(x)):
                    x[np.isnan(x)] = 0.0
                    p = np.power(x[i][0], alpha)
                    p /= p.sum()
                    next_x.append(np.random.choice(np.arange(len(x[i][0])), p=p))
                x = np.array(next_x)[:, np.newaxis]
            # store prediction
            output.append(x)
            # update target sequence
        return np.concatenate(output, -1)
