# Neural Machine Translation with Attention
To translate a sentence from a language to another one, a human translator reads the sentence part by part, and generates part of translation. A neural machine translation with attention like a human translator looks at the sentence part by part. To generate each part of translation, the attention mechanism tells a Neural Machine Translation model where it should pay attention to.
# Dataset
This notebook trains a Neural Machine Translation with Attention model that can be used to translate from one language to another, like Farsi to English translation. However, training on real-world translation data can take days of training on GPUs. To evaluate the model, I used it for date translation task. 
We will have the network learn to converts Human Readable dates like '28th of March, 1985 ' to Machine Readable format '1985-03-28'
