## Instructions

1. download and unzip the dataset:
2. set up config
3. run preprocessing
4. run training

## What I didn't do

1. additional word features
2. padding of char features before cnn component
3. cnn component hyperparams
3. softmax + weighted cross entropy loss vs log softmax plus sentence level log likelihood plus possible tag sequences
4. pretrained word embeddings
5. Adam optimizer

# Notes

1) explain task
2) explain how entities are often annotated and metrics are defined
3) explain techniques
4) explain model
5) explain code
6) explain results

- Used hyperparams from CoNLL-2003 (Round 2) NER model, excluding a few differences
- 