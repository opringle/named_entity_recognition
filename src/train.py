#modules
from collections import Counter
import mxnet as mx
import numpy as np
import sys
import os
import ast

#custom modules
import config
from misc_modules import load_obj
from data_iterators import BucketNerIter
from metrics import composite_classifier_metrics
from initializers import WeightInit

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

if config.max_training_examples:
    x_train = x_train[:config.max_training_examples]
    y_train = y_train[:config.max_training_examples]

if config.max_val_examples:
    x_test = x_test[:config.max_val_examples]
    y_test = y_test[:config.max_val_examples]

print("\ntraining sentences: ", len(x_train), "\n\ntest sentences: ", len(x_test))

#get index integer for "not entity"
not_entity_index = load_obj("../data/tag_index_dict")["O"]
print("index of 'not entity' label: ", not_entity_index)

#get counts for entities in data
train_entity_counts = Counter(entity for sublist in y_train for entity in sublist)
val_entity_counts = Counter(entity for sublist in y_test for entity in sublist)
print("\nentites in training data: ", sum(train_entity_counts.values()) - train_entity_counts[not_entity_index], "/", sum(train_entity_counts.values()))
print("entites in validation data: ", sum(val_entity_counts.values()) - val_entity_counts[not_entity_index], "/", sum(val_entity_counts.values()),"\n")

print("\nentity counts: ", train_entity_counts)

##############################
# create custom data iterators
##############################

# we want padding to use "not entity" index in labels
train_iter = BucketNerIter(sentences=x_train, 
                           entities=y_train, 
                           batch_size=config.batch_size, 
                           buckets = config.buckets,
                           data_name='seq_data',
                           label_name='seq_label',
                           label_pad=not_entity_index,
                           data_pad=-1)

val_iter = BucketNerIter(sentences=x_test,
                           entities=y_test,
                           batch_size=config.batch_size,
                           buckets=train_iter.buckets,
                           data_name='seq_data',
                           label_name='seq_label',
                           label_pad=not_entity_index,
                           data_pad=-1)

#######################
# create network symbol
#######################

#the sequential and fused cells allow stacking multiple layers of RNN cells
if config.context == mx.gpu():
    print("\n\tTRAINING ON GPU: \n")

    #the fusedrnncell is optimized for gpu computation only
    bi_cell = mx.rnn.FusedRNNCell(num_hidden=config.lstm_state_size,
                                        num_layers=config.lstm_layers,
                                        mode='lstm',
                                        bidirectional=True,
                                        dropout=config.lstm_dropout)

else:
    print("\n\tTRAINING ON CPU: \n")

    bi_cell = mx.rnn.SequentialRNNCell()

    for layer_num in range(config.lstm_layers):
        bi_cell.add(mx.rnn.BidirectionalCell(mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="forward_layer_" + str(layer_num)),
                                             mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="backward_layer_" + str(layer_num))))
        bi_cell.add(mx.rnn.DropoutCell(config.lstm_dropout))

#architecture is defined in a function, to allow variable length input sequences
def sym_gen(seq_len):
    """function that creates a network graph, depending on sequence length"""

    print("\n", "-" * 50,"\nNETWORK SYMBOL FOR SEQ LENGTH: ", seq_len, "\n", "-"*50)

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

    #initialize weight array for use in our loss function, using a custom initializer and prevent the weights from being updated
    label_weights = mx.sym.BlockGrad(data=mx.sym.Variable(shape=(1, num_labels, 1), init=WeightInit(), name='label_weights'),name = "blocked_weights")
    broadcast_label_weights = mx.sym.broadcast_to(data = label_weights, shape = (config.batch_size, num_labels, seq_len), name = 'broadcast weights')
    print("\ninput weights shape: ", broadcast_label_weights.infer_shape()[1][0])

    #create an embedding layer
    embed_layer = mx.sym.Embedding(data=seq_data, input_dim=vocab_size, output_dim=config.word_embedding_vector_length, name='vocab_embed')
    print("\nembedding layer shape: ", embed_layer.infer_shape(seq_data=input_feature_shape)[1][0])

    #unroll the lstm cell in time: output is a list of forward/backward concatenated symbols from the final layer
    bi_cell.reset()
    outputs, states = bi_cell.unroll(length=seq_len, inputs=embed_layer, merge_outputs=False)
    print("\neach of the ", seq_len, " unrolled cells has concatenated forward and backward cell shape: ", 
        outputs[0].infer_shape(seq_data=input_feature_shape)[1][0])

    #for each timestep, add a fully connected layer with num_neurons = num_possible_tags
    step_outputs = []
    for i, step_output in enumerate(outputs):
        fc = mx.sym.FullyConnected(data=step_output, num_hidden=num_labels, name = "fc_" + str(i))
        reshaped_fc = mx.sym.Reshape(data=fc, shape=(config.batch_size, num_labels, 1), name = "rfc_" + str(i))
        step_outputs.append(reshaped_fc)
    print("\nshape after each cell output passes through fully connected layer: ", reshaped_fc.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\nnumber of recurrent cell unrolls: ", len(outputs))

    #concatenate fully connected layers for each timestep
    r_output = mx.sym.concat(*step_outputs, dim=2, name = 'fc_outputs')
    print("\nshape after concatenating outputs: ", r_output.infer_shape(seq_data=input_feature_shape)[1][0])

    #apply softmax function to network output
    sm = mx.sym.softmax(data=r_output, axis=1, name='softmax_pred')
    print("\nshape after applying softmax to data: ", sm.infer_shape(seq_data=input_feature_shape)[1][0])

    #create a symbol to use with evaluation metrics, since we use a custom loss function
    softmax_output = mx.sym.BlockGrad(data = sm,name = 'softmax')

    #one hot encode label input
    one_hot_labels = mx.sym.one_hot(indices=seq_label, depth=num_labels, name='one_hot_labels')
    print("\nonehot label shape: ", one_hot_labels.infer_shape(seq_label=input_label_shape)[1][0])

    #transpose to match network output
    label = mx.sym.transpose(data=one_hot_labels, axes=(0, 2, 1), name='transposed_labels')
    print("\ntransposed onehot label shape: ", label.infer_shape(seq_label=input_label_shape)[1][0])

    #compute the cross entropy loss
    loss = -((label * mx.sym.log(sm)) + ((1 - label) * mx.sym.log(1 - sm))) * broadcast_label_weights
    print("\ncross entropy loss shape: ", loss.infer_shape(seq_data = input_label_shape, seq_label=input_label_shape)[1][0])

    #symbol to compute the gradient of the loss with respect to the input data
    loss_grad = mx.sym.MakeLoss(data = loss, name = 'loss')
    print("\nloss grad shape: ", loss_grad.infer_shape(seq_data=input_feature_shape, seq_label=input_label_shape)[1][0])

    #finally create a symbol group consisting of both the model output (gradient of loss with respect to data) and the softmax layer output (model predictions)
    network = mx.sym.Group([softmax_output, loss_grad])

    return network, ('seq_data',), ('seq_label',)


#####################################
# create a trainable bucketing module
#####################################

model = mx.mod.BucketingModule(sym_gen= sym_gen, 
                               default_bucket_key=train_iter.default_bucket_key, 
                               context = config.context)

# allocate memory given the input data and label shapes
model.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)

# initialize parameters by uniform random numbers
model.init_params(initializer=mx.init.Uniform(scale=.1))

# use SGD with learning rate 0.1 to train
model.init_optimizer(optimizer=config.optimizer, optimizer_params=config.optimizer_params)

#define a custom metric, which takes the output from an internal layer and calculates precision, recall and f1 score
metric = composite_classifier_metrics()

####################################
# fit the model to the training data
####################################

# train x epochs, i.e. going over the data iter one pass
for epoch in range(config.num_epoch):

    train_iter.reset()
    val_iter.reset()
    metric.reset()

    for batch in train_iter:

        bucket = batch.bucket_key                 #get the seq length
        model.forward(batch, is_train=True)       # compute predictions
        model.backward()                          # compute gradients
        model.update()                            # update parameters
        model.update_metric(metric, batch.label)  # accumulate metric scores on prediction module
    print('\nEpoch %d, Training %s' % (epoch, metric.get()))

    # metric.reset()

    # for batch in val_iter:
    #     bucket = batch.bucket_key
    #     model.forward(batch, is_train=False)       # compute predictions
    #     model.update_metric(metric, batch.label)   # accumulate metric scores
    # print('Epoch %d, Validation %s' % (epoch, metric.get()))

# #########################################
# # create a separate module for predicting
# #########################################

# #TODO: pred shape is not reliable when batch size not a multiple of number of training examples

# # because we are designing our own loss function, the model output symbol now returns the gradient of the loss with respect to the input data
# # to deal with this we need to create a separate module for prediction, that takes the output from an intermediate symbol

# internal_symbols = model.symbol.get_internals()  # get all internal symbols
# softmax_sym_index = internal_symbols.list_outputs().index('softmax_pred_output')  # find the index of the softmax prediction output layer
# prediction_symbol = internal_symbols[softmax_sym_index] # retrive softmax pred symbol

# # create module from internal symbol
# model_pred = mx.mod.Module(symbol=prediction_symbol,data_names=('seq_data',), label_names=None)

# # allocate memory given the input data and label shapes
# model_pred.bind(data_shapes=train_iter.provide_data, label_shapes = None)

# # initialize parameters by uniform random numbers
# model_pred.init_params(initializer=mx.init.Uniform(scale=.1))

# #set the parameters of the prediction module to the learned values
# model_pred.set_params(arg_params=model.get_params()[0], aux_params=model.get_params()[1]) # pass learned weights to prediction model


######################################################
# ensure sentences are matched with entities correctly
######################################################

# #load in dict mapping indices back to words
# word_to_index = load_obj("../data/word_index_dict")
# index_to_word = dict([(v, k) for k, v in word_to_index.items()])
# tag_to_index = load_obj("../data/tag_index_dict")
# index_to_tag = dict([(v, k) for k, v in tag_to_index.items()])

# train_iter.reset()
# for i, batch in enumerate(train_iter):
#     if i == 1:
#         data = batch.data[0].asnumpy().tolist()
#         labels = batch.label[0].asnumpy().tolist()

#         #map dict to index values to reproduce sentences
#         sentences_train = [[index_to_word[index] for index in utterance if index != -1] for utterance in data]
#         sentences_test = [[index_to_word[index] for index in utterance if index != -1] for utterance in data]
#         tags_train = [[index_to_tag[index] for index in tag_sequence if index != -1] for tag_sequence in labels]
#         tags_test = [[index_to_tag[index] for index in tag_sequence if index != -1] for tag_sequence in labels]

        # for i, sentence in enumerate(sentences_train):
        #     if i < 3:
        #         print("\nsentence>tag mapping: %d" % (i))
        #         for j in list(range(len(sentence))):
        #             print("\ntoken: ", sentences_train[i][j], "\ntag: ", tags_train[i][j])

train_iter.reset()
