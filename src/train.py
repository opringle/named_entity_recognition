#modules
import mxnet as mx
import numpy as np
import sys
import os
import ast

#custom modules
import config
from misc_modules import load_obj
from data_iterators import BucketNerIter

######################################
# load data 
######################################

with open("../data/x_train.txt") as f:
    x_train = f.readlines()
x_train = [ast.literal_eval(x.strip()) for x in x_train]

with open("../data/x_test.txt") as f:
    x_test = f.readlines()
x_test = [ast.literal_eval(x.strip()) for x in x_test]

with open("../data/y_train.txt") as f:
    y_train = f.readlines()
y_train = [ast.literal_eval(x.strip()) for x in y_train]

with open("../data/y_test.txt") as f:
    y_test = f.readlines()
y_test = [ast.literal_eval(x.strip()) for x in y_test]

x_train = x_train[:2000]
x_test = x_test[:200]
y_train = y_train[:2000]
y_test = y_test[:200]

print("\ntraining examples: ", len(x_train), "\n\ntest examples: ", len(x_test), "\n")

######################################
# create data iterators
######################################

#create custom data iterators for training and testing
train_iter = BucketNerIter(sentences=x_train, 
                           entities=y_train, 
                           batch_size=config.batch_size, 
                           buckets = config.buckets,
                           data_name='seq_data',
                           label_name='seq_label')

# val_iter = BucketNerIter(sentences=x_test,
#                            entities=y_test,
#                            batch_size=config.batch_size,
#                            buckets=config.buckets,
#                            data_name='seq_data',
#                            label_name='seq_label')


# train_iter = mx.rnn.BucketSentenceIter(sentences=x_train,
#                                        batch_size=config.batch_size,
#                                        buckets=config.buckets,
#                                        data_name='seq_data',
#                                        label_name='seq_label')


# train_iter.next()
# val_iter.next()

#show some batches to check bucket key, label and features are correct
# print(train_iter.idx)
# print(train_iter.idx[train_iter.curr_idx])
# for i in list(range(3)):
#     databatch = train_iter.next()
#     print(databatch.data, databatch.label, databatch.provide_data, databatch.provide_label)

#     if batch.bucket_key == 5:
#         print("\n\nbatch ", i, " shape: ", train_iter.provide_data)
#         print("data: ", batch.data, "\nlabel: ", batch.label, "\nbucket size: ", batch.bucket_key, )
#         break
# train_iter.reset()

######################################
# create network symbol
######################################

#create a bidirectional lstm cell
bi_cell = mx.rnn.BidirectionalCell(l_cell=mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="forward_"),
                                       r_cell=mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="backward_"))


#architecture is defined in a function, to allow variable length input sequences
def sym_gen(seq_len):
    """function that creates a network graph, depending on sequence length"""

    print("-" * 50)

    #define hyperparameters from data folder
    vocab_size = len(load_obj("../data/word_index_dict"))
    num_labels = len(load_obj("../data/tag_index_dict"))

    input_feature_shape = (config.batch_size, seq_len)
    input_label_shape = (config.batch_size, seq_len)

    #data placeholders: we are inputting a sequence of data each time.
    seq_data = mx.symbol.Variable('seq_data')
    seq_label = mx.sym.Variable('seq_label')
    print("\ninput data shape: ", seq_data.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\ninput label shape: ", seq_label.infer_shape(seq_label=input_label_shape)[1][0])

    #create an embedding layer
    embed_layer = mx.sym.Embedding(data=seq_data, input_dim=vocab_size, output_dim=config.word_embedding_vector_length, name='vocab_embed')
    print("\nembedding layer shape: ", embed_layer.infer_shape(seq_data=input_feature_shape)[1][0])

    #unroll the lstm cell in time, obtaining a concartenated symbol for each time step (forward and backwards)
    bi_cell.reset()
    outputs, states = bi_cell.unroll(length=seq_len, inputs=embed_layer, merge_outputs=False, layout="NTC")
    print("\nindividual concatenated forward and backward cell shape: ", outputs[0].infer_shape(seq_data=input_feature_shape)[1][0])

    #for each timestep, add a fully connected layer with num_neurons = num_possible_tags
    step_outputs = []
    for i, step_output in enumerate(outputs):
        fc = mx.sym.FullyConnected(data=step_output, num_hidden=num_labels)
        reshaped_fc = mx.sym.Reshape(data=fc, shape=(config.batch_size, num_labels, 1))
        step_outputs.append(reshaped_fc)
    print("\nshape after each cell output passes through fully connected layer: ", reshaped_fc.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\nnumber of recurrent cell unrolls: ", len(outputs))

    #concatenate fully connected layers for each timestep
    sm_input = mx.sym.concat(*step_outputs, dim=2)
    print("\nshape after concatenating outputs: ", sm_input.infer_shape(seq_data=input_feature_shape)[1][0])

    #transpose
    sm_input = mx.sym.transpose(sm_input, axes= (0,2,1))
    print("\ndata shape after transposing: ", sm_input.infer_shape(seq_data=input_feature_shape)[1][0])

    #reshape the label
    seq_label = mx.sym.Reshape(seq_label, shape=(-1, ))
    print("\nlabel shape after reshaping: ", seq_label.infer_shape(seq_label=input_label_shape)[1][0])

    #reshape
    sm_input = mx.sym.Reshape(sm_input, shape=(-1, num_labels))
    print("\ndata shape after reshaping: ", sm_input.infer_shape(seq_data=input_feature_shape)[1][0])

    #apply softmax cross entropy loss to each column of each training example (shape =(num_labels, tokens))
    sm = mx.sym.SoftmaxOutput(data=sm_input, label=seq_label, name='softmax')
    print("\nshape after loss function: ", sm_input.infer_shape(seq_data=input_feature_shape)[1][0])

    #set lstm pointer to back of network
    lstm = sm

    return lstm, ('seq_data',), ('seq_label',)


######################
# testing this
######################

stack = mx.rnn.SequentialRNNCell()
for i in range(1):
    stack.add(mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix='lstm_l%d_' %i))

vocab_size = len(load_obj("../data/word_index_dict"))

def sym_gen2(seq_len):
    
    print("-" * 50)
    input_feature_shape = (config.batch_size, seq_len)
    input_label_shape = (config.batch_size, seq_len)

    seq_data = mx.sym.Variable('seq_data')
    seq_label = mx.sym.Variable('seq_label')
    embed = mx.sym.Embedding(data=seq_data, input_dim=vocab_size,output_dim=config.word_embedding_vector_length, name='embed')
    print("\nembedding layer shape: ", embed.infer_shape(seq_data=input_feature_shape)[1][0])

    stack.reset()
    outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)
    print("\nrnn layer shape: ", outputs.infer_shape(seq_data=input_feature_shape)[1][0])

    pred = mx.sym.Reshape(outputs, shape=(-1, config.lstm_state_size))
    print("\nafter reshaping: ", pred.infer_shape(seq_data=input_feature_shape)[1][0])

    pred = mx.sym.FullyConnected( data=pred, num_hidden=vocab_size, name='pred')
    print("\nafter fully connect layer : ", pred.infer_shape(seq_data=input_feature_shape)[1][0])

    seq_label = mx.sym.Reshape(seq_label, shape=(-1,))
    print("\nlabel reshaping: ", seq_label.infer_shape(seq_label=input_label_shape)[1][0])

    pred = mx.sym.SoftmaxOutput(data=pred, label=seq_label, name='softmax')
    print("\nsoftmax layer shape: ", pred.infer_shape(seq_data=input_feature_shape, seq_label=input_label_shape)[1][0])

    return pred, ('seq_data',), ('seq_label',)


# create a trainable bucketing module
model = mx.mod.BucketingModule(sym_gen=sym_gen2, 
                               default_bucket_key=train_iter.default_bucket_key, 
                               context = config.context)


################
# #fit the model (not working right now)
################

model.fit(
    train_data=train_iter,
    eval_metric=mx.metric.Perplexity(-1),
    optimizer='Adam',
    optimizer_params={'learning_rate': config.learning_rate},
    initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
    num_epoch=config.num_epoch)


###############
# to debug issues
##############

# # allocate memory given the input data and label shapes
# model.bind(data_shapes=train_iter.provide_data,
#            label_shapes=train_iter.provide_label)
# # initialize parameters by uniform random numbers
# model.init_params(initializer=mx.init.Uniform(scale=.1))
# # use SGD with learning rate 0.1 to train
# model.init_optimizer(optimizer='sgd', optimizer_params=(
#     ('learning_rate', config.learning_rate), ))
# # use accuracy as the metric
# metric = mx.metric.create('acc')
# # train 5 epochs, i.e. going over the data iter one pass
# for epoch in range(config.num_epoch):
#     train_iter.reset()
#     metric.reset()
#     for batch in train_iter:
#         model.forward(batch, is_train=True)       # compute predictions
#         # accumulate prediction accuracy
#         model.update_metric(metric, batch.label)
#         model.backward()                          # compute gradients
#         model.update()                            # update parameters
#     print('Epoch %d, Training %s' % (epoch, metric.get()))

########################
# predict to check shape
########################

#print("\nmodel predictions are of shape: ", model.predict(val_iter).shape)
