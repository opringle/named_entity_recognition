## Goal

[Implenting state of the art NN architecture from this paper](https://www.aclweb.org/anthology/Q16-1026) for the task of named entity recognition in MXNet.

## Notes

- without bucketing we significantly increase the imbalance in the dataset, since we tag as "not entity"

## To do

1. train without label weighting to high accuracy
2. modify custom loss layer

    - get block grad working/make sure you can scale entity prediction losses
    - look into sentence level log-liklihood

3. modify metric to measure per class precision and recall
4. add CNN feature generation

    - modify input data (ideally still 1 data source)

5. add more features

    - spacy provides dependency, postag, shape, capitalization features
    - ensure mxnet embedding is doing what you think. we want context for vector initialization

6. Train model to high standard on [small kaggle dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
7. Prove performance on real dataset after requesting access
8. get prediction symbol working with bucketing.... needs to be a bucketing module!

## Dataset

- [Download the dataset](https://www.clips.uantwerpen.be/conll2003/ner.tgz)
- [Request access to Reuters Corpora](http://trec.nist.gov/data/reuters/reuters.html)
- Follow the instructions in the [README](https://www.clips.uantwerpen.be/conll2003/ner/000README) to generate training files

## [State of the art](https://aclweb.org/aclwiki/CONLL-2003_(State_of_the_art))


## Benefits

- demonstrates convolutional layers for feature extraction, recurrent layers allowing variable length inputs, custom classification metrics to show meaningful scores on imbalanced data, custom loss functions to handle imbalanced labels

