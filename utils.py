import tensorflow as tf
import pickle
from collections import Counter, defaultdict
from unicodedata import normalize
import re
import numpy as np
import tensorflow.keras.backend as K
import os
from tensorflow.keras.models import load_model
import math
def load_data(file):
    lines = open(file, encoding='UTF-8').read().strip().split('\n')
    sentence_pairs = []
    for line in lines:
        if '\t' not in line:
            continue

        s1, s2, _ = line.rstrip().split('\t')
        sentence_pairs.append([s1, s2])
    return sentence_pairs

def filter(sentence_pairs, Tx, Ty):
  # import pdb; pdb.set_trace()
  lengths = [ [len(s1.split()), len(s2.split())] for s1,s2 in sentence_pairs]
  good = [ True if (l1 <=Tx) and (l2 <=Ty) else False for l1,l2 in lengths]
  filtered = [s for i,s in enumerate(sentence_pairs) if good[i]]
  return filtered


def unicode_to_ascii(s):
    s = normalize('NFD', s).encode('ascii', 'ignore')
    return s.decode('UTF-8')


def clean_sentence(sentence):
    sentence = unicode_to_ascii(sentence.lower().strip())

    # creating a space between a word and the punctuation following it. Ex: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)

    sentence = sentence.rstrip().strip()
    return sentence


class LanguageVocab:
    def __init__(self, sentences):
        self.vocab = self.make_vocab(sentences)
        self.vocab.update({'<eos>', '<sos>'})
        self.word_idx = self.word_index()
        self.idx_word = self.reverse_word_index()

    def make_vocab(self, sentences, min_occurance=3):
        token_count = Counter()
        for sentence in sentences:
            tokens = sentence.split()
            token_count.update(tokens)
        print("total vocab-before triming:", len(token_count))
        vocab = [k for k, c in token_count.items() if c >= min_occurance]
        print("total vocab-after triming:", len(vocab))
        return set(vocab)

    def word_index(self):
        vocab = sorted(self.vocab)
        return dict(zip(['<pad>'] + vocab + ['<unk>'], list(range(len(vocab) + 2))))

    def reverse_word_index(self):
        return {v: k for k, v in self.word_idx.items()}


def max_length(sentences):
    lengths = [len(s.split()) for s in sentences]
    return max(lengths)


def features(sentence, language_vocab, max_length):
    tokens = sentence.split()

    tokens = [token if token in language_vocab.vocab else '<unk>' for token in tokens]

    tokens.extend(['<pad>'] * (max_length - len(tokens)))
    rep = list(map(lambda x: language_vocab.word_idx[x], tokens))
    return rep
def preprocess_sentences(sentences):

  language_vocab = LanguageVocab(sentences)
  lang_max_length = max_length(sentences)
  X = np.array([features(s,language_vocab, lang_max_length) for s in sentences])
  return X, language_vocab, lang_max_length

def save_pairs_dict(sentence_pairs):
  inp_ref_dict = defaultdict(list)
  for s1,s2 in sentence_pairs:
    inp_ref_dict[s1].append(s2)

def prepare_data(sentence_pairs, num_examples=0, Tx = 15, Ty=18):
    clean_sentence_pairs = [[clean_sentence(s1),clean_sentence(s2)]  for s1,s2 in sentence_pairs]
    clean_sentence_pairs = filter(clean_sentence_pairs, Tx, Ty)
    if num_examples > 0:
        clean_sentence_pairs = clean_sentence_pairs[0:num_examples]
    input_sentences = [s1 for s1, s2 in clean_sentence_pairs]
    target_sentences = [s2 for s1, s2 in clean_sentence_pairs]
    X, inp_vocab, inp_length = preprocess_sentences(input_sentences)
    Y, targ_vocab, targ_length = preprocess_sentences(target_sentences)
    return X, Y, inp_vocab, targ_vocab, inp_length, targ_length


def loss_func(y_train, pred):
    mask = K.cast(y_train > 0, dtype='float32')
    mask2 = tf.greater(y_train, 0)
    non_zero_y = tf.boolean_mask(pred, mask2)
    val = K.log(non_zero_y)

    return  -K.sum(val) / K.sum(mask)


def acc_func(y_train, pred):

    targ = K.argmax(y_train, axis=-1)
    pred = K.argmax(pred, axis=-1)
    correct = K.cast(K.equal(targ, pred), dtype='float32')

    mask = K.cast(K.greater(targ, 0), dtype='float32')  # filter out padding value 0.
    correctCount = K.sum(mask * correct)
    totalCount = K.sum(mask)
    return  correctCount / totalCount

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

def make_batch(X, Y, shuffle=True, batch_size=64):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    dataset = []
    batchs = math.ceil(len(X) / batch_size)
    for b in range(batchs):
        s = b * 64
        e = min(s + 64, len(X))
        dataset.append([b, X[s:e], Y[s:e]])
    return dataset

def save_result(loss, acc, loss_test, acc_test, best_acc, dir):
  f ={}
  f['loss'] = loss
  f['acc'] = acc
  f['loss_test'] = loss_test
  f['acc_test'] = acc_test
  f['best_acc'] = best_acc
  name = open(os.path.join(dir,'result.pkl'),'wb')
  pickle.dump(f,name)
  name.close()

def load_result(dir):
    pkl_file = open(os.path.join(dir, 'result.pkl'), 'rb')
    f = pickle.load(pkl_file)
    loss = f['loss']
    acc = f['acc']
    loss_test = f['loss_test']
    acc_test = f['acc_test']
    best_acc = f['best_acc']
    pkl_file.close()
    return loss, acc, loss_test, acc_test, best_acc


def save_model(model, epoch, dir):
  f = {}
  f['model'] = model
  f['epoch'] = epoch
  name = open(os.path.join(dir,'model.pkl'),'wb')
  pickle.dump(f,name)
  name.close()
  model.save(os.path.join(dir,'train_model.h5'), os.path.join(dir,'infer_model.h5'))



def load_model(dir):
  pkl_file = open(os.path.join(dir,'model.pkl'), 'rb')
  f = pickle.load(pkl_file)
  model = f['model']
  epoch = f['epoch']
  pkl_file.close()
  model.train_model = load_model(os.path.join(dir,'train_model.h5'))
  model.inference_model  = load_model(os.path.join(dir,'infer_model.h5'))
  return model, epoch