## Goal

[Implenting NN architecture from this paper](https://www.aclweb.org/anthology/Q16-1026) for the task of named entity recognition in MXNet.

## Data

- [Download the dataset](https://www.clips.uantwerpen.be/conll2003/ner.tgz)
- [Request access to Reuters Corpora](http://trec.nist.gov/data/reuters/reuters.html)
- Follow the instructions in the [README](https://www.clips.uantwerpen.be/conll2003/ner/000README) to generate training files

## Notes

- Need charachter level embeddings for each word
- Need word level embeddings for each utterance
- My model is an RNN, needs a forget gate... maybe just make two layers for transparency (forward + backward lstm cell)
- Need data so I can get this working and compare f1 score





