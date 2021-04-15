I implement encoder-decoder based seq2seq models with attention. The encoder can be a Bidirectional LSTM, a simple LSTM or GRU, and the encoder can be LSTM or GRU. I have a argument for encoder type (RNN model used in encoder); it can be 'bidirectional', 'lstm' or 'gru'. When this argument is set to 'bidirectional', the model has Bidirectional LSTM as enocder a simple LSTM as decoder. When it is set to 'lstm', the encoder and decoder are both simple LSTMs, and for the 'gru' value, they are GRUs. Thus, I can have different three models. 



## Dataset
To evaluate the models, I use English-French dataset provided by [http://www.manythings.org/anki/](http://www.manythings.org/anki/)

I computed accuracy and loss on both training and validation set on all of these three models and compared the resutls. The experiments show that the model with a Bidirectional LSTM as encoder outperforms.
## How to run

 The result shows that the first model outperms.



the encoder with different Recuurent Neural networks such as Bidirectional LSTM, simple LSTM and GRU 


# Neural Machine Translation with Attention
To translate a sentence from a language to another one, a human translator reads the sentence part by part, and generates part of translation. A neural machine translation with attention like a human translator looks at the sentence part by part. To generate each part of translation, the attention mechanism tells a Neural Machine Translation model where it should pay attention to.
# Dataset
This notebook trains a Neural Machine Translation with Attention model that can be used to translate from one language to another, like Farsi to English translation. However, training on real-world translation data can take days of training on GPUs. To evaluate the model, I used it for date translation task. 
We will have the network learn to converts Human Readable dates like '28th of March, 1985 ' to Machine Readable format '1985-03-28'

In this rep, I implemented three encoder-decoder based seq2seq models with attention. The encoder in the first one is a Bidirectional LSTM and the decoder is a simple LSTM. In the second model, the encoder and decoder are both simple LSTMs, and in the last model they are simple GRUs.
