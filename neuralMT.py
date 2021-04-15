from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
import tensorflow as tf

from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Embedding, Input, LSTM, GRU, Dense, Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from utils import softmax
import numpy as np



def lstm(units,return_sequences=False, return_state=False):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.compat.v1.keras.layers.CuDNNLSTM(units,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    recurrent_initializer='glorot_uniform')
  else:
    return LSTM(units, return_sequences=return_sequences, return_state=return_state,  recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')


def bidirectional(units,return_sequences=False, return_state=False):
    return  Bidirectional(lstm(units,return_sequences, return_state))


def gru(units, return_sequences=False, return_state=False):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.compat.v1.keras.layers.CuDNNGRU(units,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units,
                               return_sequences=return_sequences,
                               return_state=return_state,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')


rnn_archs = {'lstm': lstm, 'gru':gru, 'bidirectional':bidirectional}

class AttentionLayer:
    def __init__(self, Tx):
        self.repeator = RepeatVector(Tx)
        self.concatenator = Concatenate(axis=-1)
        self.densor1 = Dense(1024, activation="tanh")
        self.densor2 = Dense(1)
        self.activator = Activation(softmax,
                               name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
        self.dotor = Dot(axes=1)

    def one_step_attention(self,a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.

        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        """
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"
        s_prev = self.repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis
        concat = self.concatenator([a, s_prev])
        e = self.densor1(concat)
        energies = self.densor2(e)
        alphas = self.activator(energies)
        context = self.dotor([alphas, a])

        return context


class NMT_Model:

    def stack(outputs):
        outputs = K.stack(outputs)
        return K.permute_dimensions(outputs, pattern=(1, 0, 2))

    def __init__(self, rnn_arch, Tx, Ty, encoder_units, decoder_units, embedding_dim, input_vocab_size,
                 target_vocab_size, word_embedding):
        # import pdb; pdb.set_trace()
        self.rnn_arch = rnn_arch
        self.decoder_units = decoder_units

        # attention
        self.attentionLayer = AttentionLayer(Tx)

        # encoder
        self.input = Input(shape=(Tx,))
        encoder_embedding = Embedding(input_vocab_size, embedding_dim, weights=[word_embedding], input_length=Tx)
        encoder = rnn_archs[rnn_arch](encoder_units, return_sequences=True)
        encoder_inp_embedded = encoder_embedding(self.input)
        self.encoder_out = encoder(encoder_inp_embedded)

        # decoder
        self.decoder_embedding = Embedding(target_vocab_size, embedding_dim)
        if rnn_arch == 'gru':
            self.decoder = gru(units=decoder_units, return_state=True)
        else:
            self.decoder = lstm(units=decoder_units, return_state=True)
        # self.decoder = rnn_archs[rnn_arch](units=decoder_units, return_state=True)
        self.dense_decode = Dense(target_vocab_size, activation='softmax')

        # concat
        self.concat2 = Concatenate(axis=2)

        self.decoder_state_0 = Input(shape=(decoder_units,))
        if rnn_arch != 'gru':
            self.decoder_cell_0 = Input(shape=(decoder_units,))
        self.train_model = self.get_train_model(Ty)
        print('train model was built')
        self.inference_model = self.get_inference_model(Ty)
        print('inference model was built')

    def get_train_model(self, Ty):
        decoder_inp = Input(shape=(Ty,))
        decoder_inp_embedded = self.decoder_embedding(decoder_inp)

        decoder_state = self.decoder_state_0
        if self.rnn_arch != 'gru':
            decoder_cell = self.decoder_cell_0

        # Iterate attention Ty times
        outputs = []
        for t in range(Ty):

            # Get context vector with encoder and attention
            context = self.attentionLayer.one_step_attention(self.encoder_out, decoder_state)

            # For teacher forcing, get the previous word
            select_layer = Lambda(lambda x: x[:, t:t + 1])
            prevWord = select_layer(decoder_inp_embedded)

            # Concat context and previous word as decoder input

            decoder_in_concat = self.concat2([context, prevWord])

            # pass into decoder, inference output
            if self.rnn_arch == 'gru':
                pred, decoder_state = self.decoder(decoder_in_concat, initial_state=decoder_state)
            else:
                pred, decoder_state, decoder_cell = self.decoder(decoder_in_concat,
                                                                 initial_state=[decoder_state, decoder_cell])
            pred = self.dense_decode(pred)
            outputs.append(pred)

        stack_layer = Lambda(NMT_Model.stack)
        outputs = stack_layer(outputs)
        if self.rnn_arch == 'gru':
            return Model(inputs=[self.input, decoder_inp, self.decoder_state_0], outputs=outputs)
        else:
            return Model(inputs=[self.input, decoder_inp, self.decoder_state_0, self.decoder_cell_0], outputs=outputs)

    # in the inference model teacher forcing is not available
    def get_inference_model(self, Tx):
        decoder_inp = Input(shape=(1,))

        decoder_state = self.decoder_state_0

        if self.rnn_arch != 'gru':
            decoder_cell = self.decoder_cell_0

        decoder_inp_embedded = self.decoder_embedding(decoder_inp)
        # Get context vector with encoder and attention
        context = self.attentionLayer.one_step_attention(self.encoder_out, decoder_state)

        # Concat context and previous word as decoder input

        decoder_in_concat = self.concat2([context, decoder_inp_embedded])

        # pass into decoder, inference output
        if self.rnn_arch == 'gru':
            pred, decoder_state = self.decoder(decoder_in_concat, initial_state=decoder_state)
        else:
            pred, decoder_state, decoder_cell = self.decoder(decoder_in_concat,
                                                             initial_state=[decoder_state, decoder_cell])

        pred = self.dense_decode(pred)

        if self.rnn_arch == 'gru':
            return Model(inputs=[self.input, decoder_inp, self.decoder_state_0], outputs=pred)
        else:
            return Model(inputs=[self.input, decoder_inp, self.decoder_state_0, self.decoder_cell_0], outputs=pred)

    def fit(self, enc_inp, decoder_inp, targ, batch_size=64, verbose=0):

        s_0 = np.zeros((len(enc_inp), self.decoder_units))
        if self.rnn_arch != 'gru':
            c_0 = np.zeros((len(enc_inp), self.decoder_units))
            history = self.train_model.fit([enc_inp, decoder_inp, s_0, c_0], targ, batch_size=batch_size,
                                           verbose=verbose)

        else:

            history = self.train_model.fit([enc_inp, decoder_inp, s_0], targ, batch_size=batch_size, verbose=verbose)
        self.inference_model.set_weights(self.train_model.get_weights())
        return history

    def compile(self, opt, loss, metrics):
        self.train_model.compile(optimizer=opt, loss=loss, metrics=metrics)
        self.inference_model.compile(optimizer=opt, loss=loss, metrics=metrics)

    def evaluate(self, enc_inp, decoder_inp, targ, batch_size=64, verbose=0):
        s_0 = np.zeros((len(enc_inp), self.decoder_units))
        if self.rnn_arch != 'gru':
            c_0 = np.zeros((len(enc_inp), self.decoder_units))
            return self.train_model.evaluate([enc_inp, decoder_inp, s_0, c_0], targ, batch_size=batch_size,
                                             verbose=verbose)

        else:

            return self.train_model.evaluate([enc_inp, decoder_inp, s_0], targ, batch_size=batch_size, verbose=verbose)

    def inference_evaluate(self, enc_inp, decoder_inp, targ, batch_size=64, verbose=0):
        s_0 = np.zeros((len(enc_inp), self.decoder_units))
        if self.rnn_arch != 'gru':
            c_0 = np.zeros((len(enc_inp), self.decoder_units))

        Ty = targ.shape[1]
        loss = 0.0
        acc = 0.0
        preds = []

        for t in range(Ty):

            if self.rnn_arch == 'gru':
                pred = self.inference_model.predict([enc_inp, decoder_inp, s_0])
                print('prediction is done')

                loss_b, acc_b = self.inference_model.evaluate([enc_inp, decoder_inp, s_0], targ[:, t],
                                                              batch_size=batch_size,
                                                              verbose=verbose)
            else:
                pred = self.inference_model.predict([enc_inp, decoder_inp, s_0, c_0])

                loss_b, acc_b = self.inference_model.evaluate([enc_inp, decoder_inp, s_0, c_0], targ[:, t],
                                                              batch_size=batch_size, verbose=verbose)

            pred = np.argmax(pred, axis=-1)
            decoder_inp = np.expand_dims(pred, axis=1)
            preds.append(decoder_inp)
            loss += loss_b
            acc += acc_b
        return loss / Ty, acc / Ty, NMT_Model.stack(preds).numpy()

    def save(self, train_file, infer_file):
        self.train_model.save(train_file)
        self.inference_model.save(infer_file)



