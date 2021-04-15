I implement encoder-decoder based seq2seq models with attention. The encoder can be a Bidirectional LSTM, a simple LSTM or GRU, and the encoder can be LSTM or GRU. I have a argument for encoder type (RNN model used in encoder); it can be 'bidirectional', 'lstm' or 'gru'. When this argument is set to 'bidirectional', the model has Bidirectional LSTM as enocder a simple LSTM as decoder. When it is set to 'lstm', the encoder and decoder are both simple LSTMs, and for the 'gru' value, they are GRUs. Thus, I can have different three models. 



## Dataset
To evaluate the models, I use English-French dataset provided by [http://www.manythings.org/anki/](http://www.manythings.org/anki/)
## Experiment
I computed accuracy and loss on both training and validation set on all of these three models and compared the resutls. The experiments show that the model with a Bidirectional LSTM as encoder outperforms.


## Steps
### Data Preprocessing
First of all, like any other NLP task, we load the text data and perform pre-processing and also do a train-test split.


The data needs some cleaning before being used to train our neural translation model.
1. Normalizing case to lowercase.
2. Removing punctuation from each word.
3. Removing non-printable characters.
4. Converting French characters to Latin characters.
5. Removing words that contain non-alphabetic characters. 
6. Add a special token <eos> at the end of target sentences
7.  Create two dictionaries mapping from each word in vocabulary to an id, and the id to the word. 
8.  Mark all out of vocabulary (OOV) words with a special token <unk>
9. Pad each sentence to a maximum length by adding special token <pad> at the end of the sentence.
10. Convert each sentence to its feature vector:


