## Goal

[Implenting state of the art NN architecture from this paper](https://www.aclweb.org/anthology/Q16-1026) for the task of named entity recognition in MXNet.

## To do

1. fix bucketing failure when weighting loss: 

    - weights do not depend on seq length
    - they are broadcast to shape of predicted probs
    - code runs with one bucket and fails with >2 buckets (that contain data)
    - error is below

```
Check failed: assign(&dattr, (*vec)[i]) Incompatible attr in node  at 2-th input: expected (17,54), got (1,17,1)
```

2. include more features and CNN component:
    - input needs all spacy features + charachters in each token
    - iterator needs modifications to output variable length seq from input arrays
    - network graph needs modifications to use CNN on charachters
    - netowrk graph needs modifications to append CNN features + spacy features to embedding array

3. Train model to > 80% F1 score on [small kaggle dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
4. get prediction symbol working with bucketing: needs to be a bucketing module
5. Prove performance on real dataset

## Dataset

- [Download the dataset](https://www.clips.uantwerpen.be/conll2003/ner.tgz)
- [Request access to Reuters Corpora](http://trec.nist.gov/data/reuters/reuters.html)
- Follow the instructions in the [README](https://www.clips.uantwerpen.be/conll2003/ner/000README) to generate training files

## [State of the art](https://aclweb.org/aclwiki/CONLL-2003_(State_of_the_art))

###Thoughts

Could loose loss function weighting to get done...grrr
