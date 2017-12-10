#modules
import mxnet as mx
import sys
import os

#custom modules
import config


#data placeholders
input_x = mx.sym.Variable('data')
input_y = mx.sym.Variable('softmax_label')

#choose vector length for embedding each word in corpus
num_embed = config.embedding_vector_length 

#define number of unique words in training data
vocab_size = 10000

#create an embedding layer
embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')

# reshape embedded data for next layer
lstm_input = mx.sym.Reshape(data=embed_layer, target_shape=(config.batch_size, 1, config.max_utterance_length, num_embed))

#create a bidirectional lstm cell https://mxnet.incubator.apache.org/api/python/rnn.html & http://colah.github.io/posts/2015-08-Understanding-LSTMs/
bi_cell = mx.rnn.BidirectionalCell(l_cell=mx.rnn.LSTMCell(num_hidden=config.lstm_state_size),
                                   r_cell=mx.rnn.LSTMCell(num_hidden=config.lstm_state_size))

#unroll the cell forwards and backwards in time (length is number of steps to unroll, input = (batch_size,length), merge_outputs concatenates outputs accross time steps (forward and backward passes))
outputs, states = bi_cell.unroll(length=config.max_utterance_length, inputs=input_x, merge_outputs=False)

#recieve output from unrolling LSTM cell and map to a neuron per entity class
fc = mx.sym.FullyConnected(data=outputs, num_hidden= config.num_tags)

#apply softmax cross entropy loss
mlp = mx.sym.SoftmaxOutput(data=fc, name='softmax')

# create a trainable module on CPU/GPU
model = mx.mod.Module(symbol=mlp,
                        context=training_context,
                        data_names=['data'],
                        label_names=['softmax_label'])

#fit the model
model.fit(
    train_data = train_iter,
    eval_data=val_iter,
    eval_metric='f1',
    optimizer='sgd,
    optimizer_params={"learning_rate": confg.learning_rate, "epsilon": 1e-08},
    batch_end_callback=mx.callback.Speedometer(config.batch_size, 100),
    num_epoch=config.num_epoch)

