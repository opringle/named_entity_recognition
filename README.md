## Goal

[Implenting state of the art NN architecture from this paper](https://www.aclweb.org/anthology/Q16-1026) for the task of named entity recognition in MXNet.

## To do

1. get list of 2d arrays saving and loading
2. modify bucketing iterator to handle 2d numpy arrays
3. modify the network to use char features in cnn and append pos tags to embeddings
4. set hyperparameters to those in the paper
5. Train model to > 80% F1 score on [small kaggle dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
6. get prediction symbol working with bucketing: needs to be a bucketing module
7. Prove performance on real dataset

## Dataset

- [Download the dataset](https://www.clips.uantwerpen.be/conll2003/ner.tgz)
- [Request access to Reuters Corpora](http://trec.nist.gov/data/reuters/reuters.html)
- Follow the instructions in the [README](https://www.clips.uantwerpen.be/conll2003/ner/000README) to generate training files

## [State of the art](https://aclweb.org/aclwiki/CONLL-2003_(State_of_the_art))
