#!/bin/bash

for rnn_model in bidirectional
do
    echo "python -u main.py  --rnn_arch=$rnn_model" 
    python -u main.py  --rnn_arch $rnn_model  
    
done 
