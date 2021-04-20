import argparse
import os
from utils import load_data, prepare_data, save_model, save_result, make_batch, acc_func, loss_func
from neuralMT import NMT_Model

import numpy as np
from sklearn.model_selection import train_test_split
import time


rnn_arch = ['gru', 'lstm', 'bidirectional']
embed_dim = ['50','100','200','300']
parser = argparse.ArgumentParser()
parser.add_argument('--rnn_arch', '-a', metavar='RNN', default='bidirectional',
                    choices=rnn_arch,
                    help='RNN architecture: ' + ' | '.join(rnn_arch) +
                    ' (default: rbidirectional)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--hidden', default=1024, type=int, metavar='N',
                    help='number of hidden units of Recurrent Layer')
parser.add_argument('--embedding_dim', default=200, type=int, metavar='N',
                    help='dimension of embedding layer' + ' | '.join(embed_dim) +
                    ' (default: 200)')
parser.add_argument('--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--checkpoint', dest='checkpoint',
                    help='The directory used to save the trained models',
                    default='checkpoint', type=str)
parser.add_argument('--best_model_dir', dest='best_model_dir',
                    help='The directory used to save the best trained models',
                    default='best_model', type=str)
parser.add_argument('--dataset', dest='dataset',
                    help='The directory used to save the dataset',
                    default='fra.txt', type=str)
parser.add_argument('--data_dir', dest='data_dir',
                    help='The directory containing dataset',
                    default='data', type=str)
parser.add_argument('--embedding_dir', dest='embedding_dir',
                    help='The directory containing pretrained embeddings',
                    default='embedding', type=str)
parser.add_argument('--result_dir', dest='result',
                    help='The directory used to save the results',
                    default='result', type=str)
parser.add_argument('-glove', dest='glove', action='store_true',
                    help='using glove pretrained embedding')
args = parser.parse_args()
working_dir = '.'
data_dir = os.path.join(working_dir, args.data_dir )
checkpoint_dir = os.path.join(working_dir, args.checkpoint +'_'+args.rnn_arch )
if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
best_model_dir = os.path.join(working_dir, args.best_model_dir+'_'+args.rnn_arch )
if not os.path.exists(best_model_dir):
      os.makedirs(best_model_dir)
result_dir = os.path.join(working_dir, args.result+'_'+args.rnn_arch )
if not os.path.exists(result_dir):
      os.makedirs(result_dir)
embedding_dir = os.path.join(working_dir,args.embedding_dir)
embedding = 'glove.6B.'+str(args.embedding_dim)+'d.txt'






def train_one_epoch(epoch, model,X_train, Y_train):
    print("epoch:", epoch + 1)
    loss, acc, data_count = 0.0, 0.0, 0
    dataset = make_batch(X_train, Y_train, batch_size=args.batch_size)
    for batch, inp, targ in dataset:

        data_count += len(inp)

        decoder_inp = np.zeros((len(targ), Ty))
        decoder_inp[:, 1:] = targ[:, :-1]
        decoder_inp[:, 0] = targ_vocab.word_idx['<sos>']
        targ_one_hot = np.zeros((len(targ), Ty, targ_vocab_size), dtype='float32')
        for idx, tokVec in enumerate(targ):

            for tok_idx, tok in enumerate(tokVec):
                if (tok > 0):
                    targ_one_hot[idx, tok_idx, tok] = 1


        history = model.fit(inp, decoder_inp, targ_one_hot, batch_size=args.batch_size, verbose=0)

        loss_b, acc_b = history.history['loss'][0], history.history['acc_func'][0]

        loss += (loss_b * len(inp))
        acc += (acc_b * len(inp))
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                         batch,
                                                                         loss / data_count, acc / data_count))


def evaluate(model, dataset, batch_size=64, verbose=0):
    loss, acc, data_count = 0.0, 0.0, 0
    for batch, inp, targ in dataset:

        data_count += len(inp)

        decoder_inp = np.zeros((len(targ), Ty))
        decoder_inp[:, 1:] = targ[:, :-1]
        decoder_inp[:, 0] = targ_vocab.word_idx['<sos>']
        targ_one_hot = np.zeros((len(targ), Ty, targ_vocab_size), dtype='float32')
        for idx, tokVec in enumerate(targ):

            for tok_idx, tok in enumerate(tokVec):
                if (tok > 0):
                    targ_one_hot[idx, tok_idx, tok] = 1
        loss_b, acc_b = model.evaluate(inp, decoder_inp, targ_one_hot, batch_size=batch_size, verbose=verbose)
        loss += loss_b * len(inp)
        acc += acc_b * len(inp)
    return loss / data_count, acc / data_count

if __name__ == '__main__':
    sentence_pairs = load_data(os.path.join(data_dir, args.dataset))
    X, Y, inp_vocab, targ_vocab, Tx, Ty = prepare_data(sentence_pairs , num_examples= 100, Tx= 100, Ty=100)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    del X
    del Y
    inp_vocab_size = len(inp_vocab.word_idx)
    targ_vocab_size = len(targ_vocab.word_idx)
    # setting
    HIDDEN_UNITS = args.hidden
    EMBEDDING_DIM = args.embedding_dim
    encoder_units = HIDDEN_UNITS
    decoder_units = HIDDEN_UNITS




    wordVec = {}

    print('Loading wordVec')

    # load in word vectors in a dict
    word_embedding = np.zeros((inp_vocab_size, EMBEDDING_DIM))
    # if args.glove:
    #     with open(os.path.join(embedding_dir, embedding)) as f:
    #         for line in f:
    #             data = line.split()
    #             word = data[0]
    #             vec = np.asarray(data[1:], dtype='float32')
    #             wordVec[word] = vec
    #
    #     print('Finished loading wordVec.')
    #
    #
    #     # create word embedding by fetching each word vector
    #     for tok, idx in inp_vocab.word_idx.items():
    #         if idx < inp_vocab_size:
    #             word_vector = wordVec.get(tok)
    #             if word_vector is not None:
    #                 word_embedding[idx] = word_vector
    model = NMT_Model(args.rnn_arch, Tx, Ty, encoder_units, decoder_units, EMBEDDING_DIM, inp_vocab_size,
                      targ_vocab_size, word_embedding)

    model.compile(opt='adam', loss=loss_func, metrics=[acc_func])

    # final final debug
    ### debug

    EPOCHS = args.epochs


    dataset = make_batch(X_train, Y_train, batch_size=args.batch_size)
    test_dataset = make_batch(X_test, Y_test, shuffle=False, batch_size=args.batch_size)
    loss, acc = [], []
    loss_test, acc_test = [],[]
    best_acc =0.0
    for epoch in range(EPOCHS):
        start = time.time()
        train_one_epoch(epoch, model, X_train, Y_train)
        print('Time taken for 1 epoch training {} sec\n'.format(time.time() - start))
        start = time.time()
        loss_train_e, acc_train_e = evaluate(model, dataset, batch_size=args.batch_size)
        loss_test_e, acc_test_e = evaluate(model, test_dataset, batch_size=args.batch_size)

        print('Epoch {}  Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, loss_train_e, acc_train_e))
        print('Epoch {}  Loss on test {:.4f} Accuracy on test {:.4f}'.format(epoch + 1, loss_test_e, acc_test_e))

        print('Time taken for 1 epoch evaluating {} sec\n'.format(time.time() - start))
        if loss_test_e > best_acc:
            save_model(model, epoch, best_model_dir)
        loss.append(loss_train_e)
        acc.append(acc_train_e)
        loss_test.append(loss_test_e)
        acc_test.append(acc_test_e)
        save_model(model, epoch, checkpoint_dir)
        save_result(loss, acc, loss_test, acc_test, best_acc, result_dir)




