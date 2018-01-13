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
from metrics import entity_F1_score

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

x_train = x_train[:config.max_training_examples]
x_test = x_test[:config.max_val_examples]
y_train = y_train[:config.max_training_examples]
y_test = y_test[:config.max_val_examples]

print("\ntraining examples: ", len(x_train), "\n\ntest examples: ", len(x_test), "\n")

######################################
# create custom data iterators
######################################

# iterators will pad sentences and entity arrays to the bucket size
# we want padding to use "not entity" index in labels
not_entity_index = load_obj("../data/tag_index_dict")["O"]
print(not_entity_index)

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
                           buckets=config.buckets,
                           data_name='seq_data',
                           label_name='seq_label',
                           label_pad=not_entity_index,
                           data_pad=-1)

#######################
# create network symbol
#######################

#the sequential and fused cells allow stacking multiple layers of RNN cells
if config.context == mx.gpu():

    #the fusedrnncell is optimized for gpu computation only
    bi_cell = mx.rnn.FusedRNNCell(num_hidden=config.lstm_state_size,
                                        num_layers=config.lstm_layers,
                                        mode='lstm',
                                        bidirectional=True,
                                        dropout=config.lstm_dropout)

else:

    bi_cell = mx.rnn.SequentialRNNCell()

    for layer_num in range(config.lstm_layers):
        bi_cell.add(mx.rnn.BidirectionalCell(mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="forward_layer_" + str(layer_num)),
                                             mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="backward_layer_" + str(layer_num))))


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
    sm_input = mx.sym.concat(*step_outputs, dim=2, name = 'fc_outputs')
    print("\nshape after concatenating outputs: ", sm_input.infer_shape(seq_data=input_feature_shape)[1][0])

    #apply softmax cross entropy loss, preventing the gradient being computed with respect to not entity label
    sm = mx.sym.SoftmaxOutput(data=sm_input, label=seq_label, multi_output = True, name='softmax', ignore_label = not_entity_index, use_ignore = True)
    print("\nshape after loss function: ", sm_input.infer_shape(seq_data=input_feature_shape)[1][0])

    #set lstm pointer to back of network (OCD thing)
    lstm = sm

    return lstm, ('seq_data',), ('seq_label',)


#####################################
# create a trainable bucketing module
#####################################

model = mx.mod.BucketingModule(sym_gen=sym_gen, 
                               default_bucket_key=train_iter.default_bucket_key, 
                               context = config.context)

###################################
# bind & fit the module to the data
###################################

# allocate memory given the input data and label shapes
model.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)
# initialize parameters by uniform random numbers
model.init_params(initializer=mx.init.Uniform(scale=.1))
# use SGD with learning rate 0.1 to train
model.init_optimizer(optimizer=config.optimizer, optimizer_params=config.optimizer_params)
# custom metric
metric = mx.metric.CompositeEvalMetric([mx.metric.create(entity_F1_score), 
                                        mx.metric.create('loss'), 
                                        mx.metric.create('acc')])
# train x epochs, i.e. going over the data iter one pass
for epoch in range(config.num_epoch):

    train_iter.reset()
    val_iter.reset()
    metric.reset()

    for batch in train_iter:
        model.forward(batch, is_train=True)       # compute predictions
        model.update_metric(metric, batch.label)  # accumulate prediction accuracy
        model.backward()                          # compute gradients
        model.update()                            # update parameters
    print('\nEpoch %d, Training %s' % (epoch, metric.get()))

    metric.reset()

    for batch in val_iter:
        model.forward(batch, is_train=False)       # compute predictions
        model.update_metric(metric, batch.label)  # accumulate prediction accuracy
    print('Epoch %d, Validation %s' % (epoch, metric.get()))
