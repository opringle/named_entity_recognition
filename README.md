## Goal

[Implenting state of the art NN architecture from this paper](https://www.aclweb.org/anthology/Q16-1026) for the task of named entity recognition in MXNet.

## To do

1. Allow variable input length: [MXNet bucketizers](https://github.com/apache/incubator-mxnet/blob/master/example/rnn/bucketing/lstm_bucketing.py).  This requires a custom data iterator.

a. building correct symbol shapes, data iterator looks to be working
b. failing as soon as a different batch size comes in
c. mxnet examples have fixed label shape. try fixing this to see if model trains

2. Add CNN feature generation (easy)
3. Apply custom loss function: sentence level log likelihood (hard)
4. custom metrics to print during training (easy)
5. Train model on kaggle data: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus (medium)
6. Prove performance on real dataset (hard)

## Dataset

- [Download the dataset](https://www.clips.uantwerpen.be/conll2003/ner.tgz)
- [Request access to Reuters Corpora](http://trec.nist.gov/data/reuters/reuters.html)
- Follow the instructions in the [README](https://www.clips.uantwerpen.be/conll2003/ner/000README) to generate training files




