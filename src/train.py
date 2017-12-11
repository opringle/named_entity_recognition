#modules
import mxnet as mx
import numpy as np
import sys
import os

#custom modules
import config
from custom_methods import load_obj

#load numpy files
x_train = np.load("../data/x_train.npy")[:1000]
x_test = np.load("../data/x_test.npy")[:100]
y_train = np.load("../data/y_train.npy")[:1000]
y_test = np.load("../data/y_test.npy")[:100]

print("\ntraining examples: ", x_train.shape[0], "\n\ntest examples: ", x_test.shape[0])

#define hyperparameters from data folder
vocab_size = len(load_obj("../data/word_index_dict"))
max_utterance_tokens = x_train.shape[1]
num_labels = len(load_obj("../data/tag_index_dict"))

#create data iterators for training and testing
train_iter = mx.io.NDArrayIter(data={'seq_data': x_train},
                            label={'softmax_label': y_train},
                            batch_size=config.batch_size)

val_iter = mx.io.NDArrayIter(data={'seq_data': x_test},
                            label={'softmax_label': y_test},
                            batch_size=config.batch_size)

#print input shapes
print("\nfeature input shape: ", train_iter.provide_data, 
        "\n\nlabel input shape: ", train_iter.provide_label, "\n")

#data placeholders: we are inputting a sequence of data each time.
seq_input = mx.symbol.Variable('seq_data')
softmax_label = mx.sym.Variable('softmax_label')

#choose vector length for embedding each word in corpus
num_embed = config.embedding_vector_length 

#create an embedding layer
embed_layer = mx.sym.Embedding(data=seq_input, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')

#reshape embedded data for next layer (this is NTC layout)
lstm_input = mx.sym.Reshape(data=embed_layer, target_shape=(config.batch_size, max_utterance_tokens, num_embed))

#create a bidirectional lstm cell https://mxnet.incubator.apache.org/api/python/rnn.html & http://colah.github.io/posts/2015-08-Understanding-LSTMs/
bi_cell = mx.rnn.LSTMCell(num_hidden=config.lstm_state_size)
begin_state = bi_cell.begin_state()
# bi_cell = mx.rnn.BidirectionalCell(l_cell=mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix = "forward_"),
#                                    r_cell=mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix = "backward_"))

#unroll the cell forwards and backwards in time (length is number of steps to unroll, input = (batch_size,length), merge_outputs concatenates outputs accross time steps (forward and backward passes))
outputs, states = bi_cell.unroll(length=max_utterance_tokens, inputs=lstm_input, merge_outputs=True, layout = "NTC")

#recieve output from unrolling LSTM cell and map to a neuron per entity class
fc = mx.sym.FullyConnected(data=outputs, num_hidden= max_utterance_tokens)

#apply softmax cross entropy loss
mlp = mx.sym.SoftmaxOutput(data=fc, name='softmax')

# Visualize your network
mx.viz.plot_network(mlp, save_format='png', title = "../images/network.png")

# create a trainable module on CPU/GPU
model = mx.mod.Module(symbol=mlp,
                        data_names=['seq_data'],
                        label_names=['softmax_label'])

###############
# #debug issues
##############

# # allocate memory given the input data and label shapes
# model.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)
# # initialize parameters by uniform random numbers
# model.init_params(initializer=mx.init.Uniform(scale=.1))
# # use SGD with learning rate 0.1 to train
# model.init_optimizer(optimizer='sgd', optimizer_params=(
#     ('learning_rate', 0.1), ))
# # use accuracy as the metric
# metric = mx.metric.create('acc')
# # train 5 epochs, i.e. going over the data iter one pass
# for epoch in range(5):
#     train_iter.reset()
#     metric.reset()
#     for batch in train_iter:
#         model.forward(batch, is_train=True)       # compute predictions
#         # accumulate prediction accuracy
#         model.update_metric(metric, batch.label)
#         model.backward()                          # compute gradients
#         model.update()                            # update parameters
#     print('Epoch %d, Training %s' % (epoch, metric.get()))

################
# #fit the model
################

model.fit(
    train_data = train_iter,
    eval_data=val_iter,
    eval_metric='accuracy',
    optimizer='sgd',
    optimizer_params={"learning_rate": config.learning_rate},
    batch_end_callback=mx.callback.Speedometer(config.batch_size, 100),
    num_epoch=config.num_epoch)

