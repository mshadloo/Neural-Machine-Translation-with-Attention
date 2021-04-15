I implement encoder-decoder based seq2seq models with attention. The encoder can be a Bidirectional LSTM, a simple LSTM or GRU, and the decoder can be LSTM or GRU. I have a argument for encoder type (RNN model used in encoder); it can be 'bidirectional', 'lstm' or 'gru'. When this argument is set to 'bidirectional', the model has Bidirectional LSTM as enocder a simple LSTM as decoder. When it is set to 'lstm', the encoder and decoder are both simple LSTMs, and for the 'gru' value, they are GRUs. Thus, I can have different three models. A simple encoder-decoder model without the attention mechanism tends to forget the earlier part of the sequence once they process further. With the attention mechanism, the model can deal with long sequences. To generate each part of translation, the attention mechanism tells a Neural Machine Translation model where it should pay attention to.



## Dataset
To evaluate the models, I use English-French dataset provided by [http://www.manythings.org/anki/](http://www.manythings.org/anki/)
## Experiment
I computed accuracy and loss on both training and validation set on all of these three models and compared the resutls. The experiments show that the model with a Bidirectional LSTM as the encoder outperforms the rest.

NMT with Bidirectional LSTM has lowest loss during 20 epochs          |  NMT with Bidirectional LSTM has highest accuracy during 20 epochs 
:-------------------------:|:-------------------------:
![](/images/loss.png)  |  ![](/images/accuracy.png)


 ### How to run:
```
git clone https://github.com/mshadloo/Neural-Machine-Translation-with-Attention.git
cd Neural-Machine-Translation-with-Attention
chmod +x data.sh && ./data.sh
chmod +x run.sh && ./run.sh
```
# Steps
### Data Preprocessing
First of all, like any other NLP task, we load the text data and perform pre-processing and also do a train-test split.


The data needs some cleaning before being used to train our neural translation model.
1. Normalizing case to lowercase.
2. Removing punctuation from each word.
3. Removing non-printable characters.
4. Converting French characters to Latin characters.
5. Removing words that contain non-alphabetic characters. 
6. Add a special token \<eos\> at the end of target sentences
7.  Create two dictionaries mapping from each word in vocabulary to an id, and the id to the word. 
8.  Mark all out of vocabulary (OOV) words with a special token \<unk\>
9. Pad each sentence to a maximum length by adding special token \<pad\> at the end of the sentence.
10. Convert each sentence to its feature vector

### Define The Model
I implement encoder-decoder based seq2seq models with attention. The encoder and the decoder are pre-attention and post-attention RNNs on both sides of the attention mechanism.
* Encoder:a RNN (Bidirectional LSTM, LSTM, GRU)
  * The encoder goes through 𝑇𝑥 time steps (𝑇𝑥: maximum length of the input sequence). 
* Decoder: a RNN (LSTM, GRU)
   * The decoder goes through 𝑇𝑦 time steps (𝑇𝑦: maximum length of the output sequence). 
* The attention mechanism computes the context variable  𝑐𝑜𝑛𝑡𝑒𝑥𝑡⟨𝑡⟩  for each timestep in the output ( 𝑡=1,…,𝑇𝑦 ).

