import mxnet as mx
import bisect
import random
import numpy as np
from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet import ndarray
from sklearn.utils import shuffle


#TODO: #provide data and provide label not reactive!
class BucketNerIter(DataIter):
    """This iterator can handle variable length feature/label arrays for MXNet RNN classifiers"""

    def __init__(self, sentences, entities, batch_size, buckets=None, data_pad=-1, label_pad = -1,
                 data_name='data', label_name='softmax_label', dtype='float32',
                 layout='NT'):

        super(BucketNerIter, self).__init__()

        #if buckets are not defined, create a bucket for every seq length where there are more examples than the batch size
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in sentences])) if j >= batch_size]
        buckets.sort()
        thing = np.bincount([len(s) for s in sentences])
        keys = [str(index) for index, len_count in enumerate(thing)]
        values = [len_count for index, len_count in enumerate(thing)]
        dictionary = dict(zip(keys, values))

        # print("\n\tDATA PER BUCKET: \n")
        # for bucket in buckets:
        #     print(bucket, ":", dictionary[str(bucket)])
        
        #make sure buckets have been defined
        assert (len(buckets) > 0), "no buckets could be created, not enough utterances of a certain length to create a bucket"

        ndiscard = 0

        #create empty nested lists for storing data that falls into each bucket
        self.data = [[] for _ in buckets]

        #loop through sentences
        for i, tokenized_sentence in enumerate(sentences):

            #find the index of the smallest bucket that is larger than the sentence length
            buck = bisect.bisect_left(buckets, len(tokenized_sentence))
            #if the sentence is larger than the largest bucket, discard it
            if buck == len(buckets):
                ndiscard += 1
                continue
            #create an array of shape (bucket_size,) filled with 'data_pad'
            buff = np.full((buckets[buck],), data_pad, dtype=dtype)
            #replace elements up to the sentence length with actual values
            # eg. sent length=8, bucket size=10: [1,3,4,5,6,3,8,7,-1,-1]
            buff[:len(tokenized_sentence)] = tokenized_sentence
            #append array to index = bucket index
            self.data[buck].append(buff)

        #convert to list of array of array
        self.data = [np.asarray(i, dtype=dtype) for i in self.data]

        self.label = [[] for _ in buckets]

        #loop through entities
        for i, entity_list in enumerate(entities):

            #find the index of the smallest bucket that is larger than the sentence length
            buck = bisect.bisect_left(buckets, len(entity_list))
            #if the sentence is larger than the largest bucket, discard it
            if buck == len(buckets):
                ndiscard += 1
                continue
            #create an array of shape (bucket_size,) filled with 'label_pad'
            buff = np.full((buckets[buck],), label_pad, dtype=dtype)
            #replace elements up to the sentence length with actual values
            # eg. sent length=8, bucket size=10: [1,3,4,5,6,3,8,7,-1,-1]
            buff[:len(entity_list)] = entity_list
            #append array to index = bucket index
            self.label[buck].append(buff)

        #convert to list of array of array
        self.label = [np.asarray(i, dtype=dtype) for i in self.label]

        #print("WARNING: discarded %d utterances longer than the largest bucket." % ndiscard)

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.data_pad = data_pad
        self.label_pad = label_pad
        self.nddata = []
        self.ndlabel = []
        self.major_axis = layout.find('N')
        self.layout = layout
        self.default_bucket_key = max(buckets)

        #define provide data/label based on chosen layout
        if self.major_axis == 0:
            self.provide_data = [DataDesc(name=self.data_name, shape=(
                batch_size, self.default_bucket_key), layout=self.layout)]
            self.provide_label = [DataDesc(name=self.label_name, shape=(
                batch_size, self.default_bucket_key), layout=self.layout)]
        elif self.major_axis == 1:
            self.provide_data = [DataDesc(name=self.data_name, shape=(
                self.default_bucket_key, batch_size), layout=self.layout)]
            self.provide_label = [DataDesc(name=self.label_name, shape=(
                self.default_bucket_key, batch_size), layout=self.layout)]
        else:
            raise ValueError(
                "Invalid layout %s: Must by NT (batch major) or TN (time major)")

        #create empty list to store batch index values
        self.idx = []
        #for each bucketarray
        for i, buck in enumerate(self.data):
            #extend the list eg output with batch size 5 and 20 training examples in bucket. [(0,0), (0,5), (0,10), (0,15), (1,0), (1,5), (1,10), (1,15)]
            self.idx.extend([(i, j) for j in range(
                0, len(buck) - batch_size + 1, batch_size)])
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

        if self.major_axis == 1:
            data = self.nddata[i][j:j + self.batch_size].T
            label = self.ndlabel[i][j:j + self.batch_size].T
        else:
            data = self.nddata[i][j:j + self.batch_size]
            label = self.ndlabel[i][j:j + self.batch_size]

        return DataBatch([data], [label], pad=0,
                         bucket_key=self.buckets[i],
                         provide_data=[DataDesc(name=self.data_name, shape=data.shape, layout=self.layout)],
                         provide_label=[DataDesc(name=self.label_name, shape=label.shape, layout=self.layout)])
        
