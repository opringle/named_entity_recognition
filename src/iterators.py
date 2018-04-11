import mxnet as mx
import bisect
import random
import numpy as np
from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet import ndarray
from sklearn.utils import shuffle

class BucketNerIter(DataIter):
    """
    This iterator can handle variable length feature/label arrays for MXNet RNN classifiers.
    This iterator can ingest 2d list of sentences, 2d list of entities and 3d list of characters.

    """

    def __init__(self, sentences, entities, batch_size, buckets=None, data_pad=-1, label_pad = -1,
                 data_name='data', label_name='softmax_label', dtype='float32'
                 ):

        super(BucketNerIter, self).__init__()

        #if buckets are not defined, create a bucket for every seq length where there are more examples than the batch size
        if not buckets:
            seq_counts = np.bincount([len(s) for s in entities])
            buckets = [i for i, j in enumerate(seq_counts) if j >= batch_size]
        buckets.sort()

        #make sure buckets have been defined
        assert (len(buckets) > 0), "no buckets could be created, not enough utterances of a certain length to create a bucket"

        nslice = 0

        #create empty nested lists for storing data that falls into each bucket
        self.data = [[] for _ in buckets]

        #loop through list of feature arrays
        features = sentences[0].shape[0]
        for i, feature_array in enumerate(sentences):

            #find the index of the smallest bucket that is larger than the sentence length
            buck = bisect.bisect_left(buckets, feature_array.shape[1])

            #if the sentence is larger than the largest bucket, slice it
            if buck == len(buckets):

                #set index back to largest bucket
                buck = buck - 1
                nslice += 1
                feature_array = feature_array[:, :buckets[buck]]

            #create an array of shape (features, bucket_size) filled with 'data_pad'
            buff = np.full((features, buckets[buck]), data_pad, dtype=dtype)

            #replace elements up to the sentence length with actual values
            buff[:, :feature_array.shape[1]] = feature_array

            #append array to index = bucket index
            self.data[buck].append(buff)

        #convert to list of array of 2d array
        self.data = [np.asarray(i, dtype=dtype) for i in self.data]

        self.label = [[] for _ in buckets]

        #loop through tag arrays
        for i, tag_array in enumerate(entities):

            #find the index of the smallest bucket that is larger than the sentence length
            buck = bisect.bisect_left(buckets, len(tag_array))

            #if the sentence is larger than the largest bucket, discard it
            if buck == len(buckets):

                #set index back to largest bucket
                buck = buck - 1
                nslice += 1
                tag_array = tag_array[:buckets[buck]]

            #create an array of shape (bucket_size,) filled with 'label_pad'
            buff = np.full((buckets[buck],), label_pad, dtype=dtype)

            #replace elements up to the sentence length with actual values
            buff[:len(tag_array)] = tag_array

            #append array to index = bucket index
            self.label[buck].append(buff)

        #convert to list of array of array
        self.label = [np.asarray(i, dtype=dtype) for i in self.label]

        print("WARNING: sliced %d utterances because they were longer than the largest bucket." % nslice)

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.data_pad = data_pad
        self.label_pad = label_pad
        self.nddata = []
        self.ndlabel = []
        self.default_bucket_key = max(buckets)
        self.layout = 'NT'

        #define provide data/label
        self.provide_data = [DataDesc(name=self.data_name, shape=(batch_size, features, self.default_bucket_key), layout=self.layout)]
        self.provide_label = [DataDesc(name=self.label_name, shape=(batch_size, self.default_bucket_key), layout=self.layout)]

        #create empty list to store batch index values
        self.idx = []

        #for each bucketarray
        for i, buck in enumerate(self.data):

            #extend the list eg output with batch size 5 and 20 training examples in bucket. [(0,0), (0,5), (0,10), (0,15), (1,0), (1,5), (1,10), (1,15)]
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.curr_idx = 0
        #shuffle data in each bucket
        random.shuffle(self.idx)
        for i, buck in enumerate(self.data):
            self.data[i], self.label[i] = shuffle(self.data[i], self.label[i])

        self.nddata = []
        self.ndlabel = []

        #for each bucket of data
        for buck in self.data:
            #append the data list with the data array
            self.nddata.append(ndarray.array(buck, dtype=self.dtype))
        for buck in self.label:
            #append the label list with an array
            self.ndlabel.append(ndarray.array(buck, dtype=self.dtype))

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx == len(self.idx):
            raise StopIteration
        #i = batches index, j = starting record
        i, j = self.idx[self.curr_idx] 
        self.curr_idx += 1


        data = self.nddata[i][j:j + self.batch_size]
        label = self.ndlabel[i][j:j + self.batch_size]


        return DataBatch([data], [label], pad=0,
                         bucket_key=self.buckets[i],
                         provide_data=[DataDesc(name=self.data_name, shape=data.shape, layout=self.layout)],
                         provide_label=[DataDesc(name=self.label_name, shape=label.shape, layout=self.layout)])
        
