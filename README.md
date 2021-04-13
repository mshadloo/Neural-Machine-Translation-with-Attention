In this rep, I implemented three encoder-decoder based seq2seq models with attention. The encoder in the first one is a Bidirectional LSTM and the decoder is a simple LSTM. In the second model, the encoder and decoder are both simple LSTMs, and in the last model they are simple GRUs.

## Dataset
I used English-French dataset provided by [http://www.manythings.org/anki/](http://www.manythings.org/anki/).

I computed accuracy and loss in both training and validation set on all of these models and compared the resutls. The result shows that the first model outperms.



the encoder with different Recuurent Neural networks such as Bidirectional LSTM, simple LSTM and GRU 


# Neural Machine Translation with Attention
To translate a sentence from a language to another one, a human translator reads the sentence part by part, and generates part of translation. A neural machine translation with attention like a human translator looks at the sentence part by part. To generate each part of translation, the attention mechanism tells a Neural Machine Translation model where it should pay attention to.
# Dataset
This notebook trains a Neural Machine Translation with Attention model that can be used to translate from one language to another, like Farsi to English translation. However, training on real-world translation data can take days of training on GPUs. To evaluate the model, I used it for date translation task. 
We will have the network learn to converts Human Readable dates like '28th of March, 1985 ' to Machine Readable format '1985-03-28'
