## Goal

[Implenting state of the art NN architecture from this paper](https://www.aclweb.org/anthology/Q16-1026) for the task of named entity recognition in MXNet.

## To do

1. include more features and CNN component:
    - input needs all spacy features + characters in each token
    - iterator needs modifications to output variable length seq from input arrays
    - network graph needs modifications to use CNN on characters
    - network graph needs modifications to append CNN features + spacy features to embedding array

2. Train model to > 80% F1 score on [small kaggle dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
3. get prediction symbol working with bucketing: needs to be a bucketing module
4. Prove performance on real dataset

## Dataset

- [Download the dataset](https://www.clips.uantwerpen.be/conll2003/ner.tgz)
- [Request access to Reuters Corpora](http://trec.nist.gov/data/reuters/reuters.html)
- Follow the instructions in the [README](https://www.clips.uantwerpen.be/conll2003/ner/000README) to generate training files

## [State of the art](https://aclweb.org/aclwiki/CONLL-2003_(State_of_the_art))
