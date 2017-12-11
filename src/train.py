#modules
import mxnet as mx
import numpy as np
import sys
import os

#custom modules
import config
from custom_methods import load_obj

#load numpy files
x_train = np.load("../data/x_train.npy")[:2000]
x_test = np.load("../data/x_test.npy")[:200]
y_train = np.load("../data/y_train.npy")[:2000]
y_test = np.load("../data/y_test.npy")[:200]

print("\ntraining examples: ",
      x_train.shape[0], "\n\ntest examples: ", x_test.shape[0])

#define hyperparameters from data folder
vocab_size = len(load_obj("../data/word_index_dict"))
max_utterance_tokens = x_train.shape[1]
num_labels = len(load_obj("../data/tag_index_dict"))

#create data iterators for training and testing
train_iter = mx.io.NDArrayIter(data={'seq_data': x_train},
                               label={'seq_label': y_train},
                               batch_size=config.batch_size)

val_iter = mx.io.NDArrayIter(data={'seq_data': x_test},
                             label={'seq_label': y_test},
                             batch_size=config.batch_size)

#print input shapes
print("\nfeature input shape: ", train_iter.provide_data,
      "\n\nlabel input shape: ", train_iter.provide_label, "\n")

#data placeholders: we are inputting a sequence of data each time.
seq_data = mx.symbol.Variable('seq_data')
seq_label = mx.sym.Variable('seq_label')

#create an embedding layer
embed_layer = mx.sym.Embedding(data=seq_data, input_dim=vocab_size,
                               output_dim=config.embedding_vector_length, name='vocab_embed')

#reshape embedded data for next layer (this is NTC layout)
lstm_input = mx.sym.Reshape(data=embed_layer, target_shape=(
    config.batch_size, max_utterance_tokens, config.embedding_vector_length))

#create a bidirectional lstm cell https://mxnet.incubator.apache.org/api/python/rnn.html & http://colah.github.io/posts/2015-08-Understanding-LSTMs/
bi_cell = mx.rnn.BidirectionalCell(l_cell=mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="forward_"),
                                   r_cell=mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="backward_"))

#unroll the lstm cell in time, obtaining a symbol for each time step forward and backwards
# Each symbol is of shape (batch_size, hidden_dim)
outputs, states = bi_cell.unroll(
    length=max_utterance_tokens, inputs=lstm_input, merge_outputs=False, layout="NTC")

#for each timestep, add a fully connected layer with num_neurons = num_possible_tags
step_outputs = []
for i, step_output in enumerate(outputs):
    fc = mx.sym.FullyConnected(data=step_output, num_hidden=num_labels)
    step_outputs.append(fc)

#concatenate fully connected layers for each timestep
concat = mx.sym.concat(*step_outputs, dim=1)

#reshape before applying softmax loss
sm_input = mx.sym.Reshape(data=concat, target_shape=(
    config.batch_size, num_labels, max_utterance_tokens))

#apply softmax cross entropy loss to each column of each training example (shape =(num_labels, tokens))
sm = mx.sym.SoftmaxOutput(data=sm_input, label=seq_label,
                          name='softmax', multi_output=True)

#set lstm pointer to back of network
lstm = sm

# Visualize the network
mx.viz.plot_network(lstm, save_format='png', title="../images/network.png")

# create a trainable module on CPU/GPU
model = mx.mod.Module(symbol=lstm,
                      context=mx.cpu(),
                      data_names=['seq_data'],
                      label_names=['seq_label'])


################
# #fit the model (not working right now)
################

model.fit(
    train_data=train_iter,
    eval_data=val_iter,
    eval_metric='accuracy',
    optimizer='sgd',
    optimizer_params={"learning_rate": config.learning_rate},
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
