import mxnet as mx
import random
import sys
import random

#custom modules
from data_iterators import BucketNerIter
from initializers import WeightInit
from metrics import composite_classifier_metrics

breakit = True
if breakit == True:
    random.seed(0)
else:
    random.seed(4)

#synthetically create 1000 encoded sentences
x_train = []
y_train = []
for i in list(range(1000)):
    sentence_length = random.randint(1, 20)
    sentence = random.sample(range(20), sentence_length)
    tags = random.sample(range(20), sentence_length)
    x_train.append(sentence)
    y_train.append(tags)

num_labels = 20

#define hyperparameters/info
vocab = list(set(index for list in x_train for index in list))
batch_size = 10
num_embed = 6
num_hidden = 1
buckets = [3,4,5,6,7,20]
epochs = 5

# we want padding to use "not entity" index in labels
train_iter = BucketNerIter(sentences=x_train,
                           entities=y_train,
                           batch_size=batch_size,
                           buckets=buckets,
                           data_name='seq_data',
                           label_name='seq_label',
                           label_pad=15,
                           data_pad=-1)

print("training iterator buckets: ", train_iter.buckets)
for i, batch in enumerate(train_iter):
    print("bucket size: ", batch.bucket_key, "batch shape: ", batch.data[0].shape, "batch bucket key: ", batch.bucket_key)

#define recurrent cell
r_cell = mx.rnn.LSTMCell(num_hidden=num_hidden)

def sym_gen(seq_len):
    """function that creates a network graph, depending on sequence length"""

    print("\n", "-" * 50,"\nNETWORK SYMBOL FOR SEQ LENGTH: ", seq_len, "\n", "-"*50)

    #define data shapes
    data_shape = (batch_size, seq_len)
    label_shape = (batch_size, seq_len)

    seq_data = mx.sym.Variable('seq_data')
    print("\ndata shape: ", seq_data.infer_shape(seq_data=data_shape)[1][0])

    seq_label = mx.sym.Variable('seq_label')
    print("\nlabel shape: ", seq_label.infer_shape(seq_label=label_shape)[1][0])

    weights = mx.sym.BlockGrad(mx.sym.Variable(shape=(1, 1, num_labels), init=WeightInit(), name='weights'))
    print("\ninput weights shape: ", weights.infer_shape()[1][0])

    bc_weights = mx.sym.broadcast_to(weights, shape = (batch_size, seq_len, num_labels), name = 'broadcast_weights')
    print("\nbroadcast input weights shape: ", bc_weights.infer_shape()[1][0])

    embed = mx.sym.Embedding(seq_data, input_dim=len(vocab), output_dim=num_embed,name='embed')
    print("\nembed layer shape: ", embed.infer_shape(seq_data=data_shape)[1][0])

    output, shapes = r_cell.unroll(seq_len, inputs=embed, merge_outputs=True)
    print("\nconcatenated recurrent layer shape: ", output.infer_shape(seq_data=data_shape)[1][0], "after ", seq_len, " unrolls")

    output = mx.sym.Reshape(output, shape=(-1,1))
    print("\nreshaped output shape: ", output.infer_shape(seq_data=data_shape)[1][0])

    fc = mx.sym.FullyConnected(output, num_hidden=num_labels, name='pred')
    print("\nfully connected layer shape: ", fc.infer_shape(seq_data=data_shape)[1][0])

    fc = mx.sym.Reshape(fc, shape = (batch_size, seq_len, num_labels))
    print("\nreshaped fully connected layer shape: ", fc.infer_shape(seq_data=data_shape)[1][0])

    sm = mx.sym.softmax(fc, name='predicted_proba')
    print("\nsoftmax shape: ", sm.infer_shape(seq_data=data_shape)[1][0])

    softmax_output = mx.sym.BlockGrad(data=sm, name='softmax')

    onehot_label = mx.sym.one_hot(indices=seq_label, depth=num_labels, name='one_hot_labels')
    print("\none hot label shape: ", onehot_label.infer_shape(seq_label=label_shape)[1][0])

    loss = -((onehot_label * mx.sym.log(sm)) + ((1 - onehot_label) * mx.sym.log(1 - sm))) * bc_weights

    loss_grad = mx.sym.MakeLoss(loss, name = 'loss')
    print("\nloss grad shape: ", loss_grad.infer_shape(seq_data=data_shape, seq_label=label_shape)[1][0])

    network = mx.sym.Group([softmax_output, loss_grad])

    return network, ('seq_data',), ('seq_label',)

#create a trainable bucketing module on cpu
model = mx.mod.BucketingModule(sym_gen=sym_gen, default_bucket_key=train_iter.default_bucket_key, context=mx.cpu())

#fit the module
metric = composite_classifier_metrics()
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
