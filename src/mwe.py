import mxnet as mx
import random
import sys

#parameter to break this code or not
break_code = sys.argv[1].lower() == "true"

#synthetically create 1000 encoded sentences
encoded_sentences = []
for i in list(range(1000)):
    sentence_length = random.randint(1, 20)
    sentence = random.sample(range(20), sentence_length)
    encoded_sentences.append(sentence)

#define hyperparameters/info
vocab = list(set(index for list in encoded_sentences for index in list))
batch_size = 18
num_embed = 10
num_hidden = 3
buckets = [5, 10, 15, 20]
epochs = 15

# Create and register a custom initializer
@mx.init.register
class WeightInit(mx.init.Initializer):
    def __init__(self):
        super(WeightInit, self).__init__()
    def _init_weight(self, _, arr):
        #set all weights to 1
        arr[:] = 1
        #scale the weight for not entity values
        arr[:,5,:] /= 5
    def _init_bias(self, _, arr):
        arr[:] =1

#use mxnet bucketing iterator, label is next value in sentence
train_iter = mx.rnn.BucketSentenceIter(sentences=encoded_sentences,
                                       batch_size=batch_size,
                                       buckets=buckets,
                                       data_name='data',
                                       label_name='softmax_label')

#define recurrent cell
r_cell = mx.rnn.LSTMCell(num_hidden=num_hidden)

#architecture is defined in a function, to allow variable length input sequences
def sym_gen(seq_len):
    """function that creates a network graph, depending on sequence length"""

    #define data shapes
    data_shape = (batch_size, seq_len)
    label_shape = (batch_size, seq_len)

    data = mx.sym.Variable('data')
    print("\ndata shape: ", data.infer_shape(data=data_shape)[1][0], "\n")

    label = mx.sym.Variable('softmax_label')
    print("\nlabel shape: ", label.infer_shape(softmax_label=label_shape)[1][0], "\n")

    #initialize weight array for use in our loss function, using a custom initializer and prevent the weights from being updated
    label_weights = mx.sym.BlockGrad(mx.sym.Variable(shape=(1, len(vocab), 1), init=WeightInit(), name='label_weights'),name = "blocked_weights")
    broadcast_label_weights = mx.sym.broadcast_to(label_weights, shape = (batch_size, len(vocab), seq_len), name = 'broadcast_weights')
    print("\ninput weights shape: ", broadcast_label_weights.infer_shape()[1][0])

    embed = mx.sym.Embedding(data, input_dim=len(vocab), output_dim=num_embed,name='embed')
    print("\nembed layer shape: ", embed.infer_shape(data=data_shape)[1][0], "\n")

    output, shapes = r_cell.unroll(seq_len, inputs=embed, merge_outputs=True)
    print("\nconcatenated recurrent layer shape: ", output.infer_shape(data=data_shape)[1][0], "after ", seq_len, " unrolls\n")

    reshape = mx.sym.Reshape(output, shape=(-1, num_hidden))
    print("\nafter reshaping: ", reshape.infer_shape(data=data_shape)[1][0], "\n")

    fc = mx.sym.FullyConnected(reshape, num_hidden=len(vocab), name='pred')
    print("\nfully connected layer shape: ", fc.infer_shape(data=data_shape)[1][0], "\n")

    label = mx.sym.Reshape(label, shape=(-1,))
    print("\nlabel shape after reshaping: ", label.infer_shape(softmax_label=label_shape)[1][0], "\n")

    onehot_label = mx.sym.one_hot(indices=label, depth=len(vocab), name='one_hot_labels')
    print("\none hot label shape: ", onehot_label.infer_shape(softmax_label=label_shape)[1][0], "\n")

    sm = mx.sym.softmax(fc, name='predicted_proba')
    print("\nsoftmax shape: ", sm.infer_shape(data=data_shape)[1][0], "\n")

    if break_code == False:
        loss = -((onehot_label * mx.sym.log(sm)) + ((1 - onehot_label) * mx.sym.log(1 - sm)))
    else:
        loss = -((onehot_label * mx.sym.log(sm)) + ((1 - onehot_label) * mx.sym.log(1 - sm))) * broadcast_label_weights

    print("\ncross entropy loss shape: ", loss.infer_shape(data = data_shape, softmax_label=label_shape)[1][0])

    loss_grad = mx.sym.MakeLoss(loss, name = 'loss')
    print("\nloss grad shape: ", loss_grad.infer_shape(data=data_shape, softmax_label=label_shape)[1][0])

    return loss_grad, ('data',), ('softmax_label',)

#create a trainable bucketing module on cpu
model = mx.mod.BucketingModule(sym_gen=sym_gen, default_bucket_key=train_iter.default_bucket_key, context=mx.cpu())

#fit the module
metric = mx.metric.create('loss')
model.bind(data_shapes=train_iter.provide_data,
           label_shapes=train_iter.provide_label)
model.init_params()
model.init_optimizer(optimizer='Adam')
for epoch in range(epochs):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        model.forward(batch, is_train=True)             # compute predictions
        model.backward()                                # compute gradients
        model.update()                                  # update parameters
        model.update_metric(metric, batch.label)        # update metric
    print('\n', 'Epoch %d, Training %s' % (epoch, metric.get()))
