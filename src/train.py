#modules
from collections import Counter
import mxnet as mx
import numpy as np
import sys
import os
import ast

#custom modules
import config
from data_helpers import load_obj
from iterators import BucketNerIter
from metrics import composite_classifier_metrics
from initializers import WeightInit

#########################
# load and summarize data 
#########################

x_train = load_obj("../data/x_train")
y_train = load_obj("../data/y_train")
x_test = load_obj("../data/x_test")
y_test = load_obj("../data/y_test")

if config.max_training_examples:
    x_train = x_train[:config.max_training_examples]
    y_train = y_train[:config.max_training_examples]

if config.max_val_examples:
    x_test = x_test[:config.max_val_examples]
    y_test = y_test[:config.max_val_examples]

print("\ntraining sentences: {} \ntest sentences {}".format(len(x_train), len(x_test)))

#infer dataset features required to build module
outside_tag_index = load_obj("../data/tag_to_index")["O"]
num_labels = len(load_obj("../data/tag_to_index"))
vocab_size = len(load_obj("../data/word_to_index"))
tag_vocab_size = len(load_obj("../data/pos_to_index"))
char_vocab_size = len(load_obj("../data/char_to_index"))
features = x_train[0].shape[0]

#get counts for entities in data
train_entity_counts = Counter(entity for sublist in y_train for entity in sublist)
val_entity_counts = Counter(entity for sublist in y_test for entity in sublist)
print("\nentites in training data: {} / {}".format(sum(train_entity_counts.values()) - train_entity_counts[outside_tag_index], 
                                                   sum(train_entity_counts.values())))
print("entites in validation data: {} / {}".format(sum(val_entity_counts.values()) - val_entity_counts[outside_tag_index], 
                                                   sum(val_entity_counts.values())))

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
                           label_pad=outside_tag_index,
                           data_pad=-1)

val_iter = BucketNerIter(sentences=x_test,
                           entities=y_test,
                           batch_size=config.batch_size,
                           buckets=train_iter.buckets,
                           data_name='seq_data',
                           label_name='seq_label',
                           label_pad=outside_tag_index,
                           data_pad=-1)

#######################
# iterator sanity check
#######################

index_to_tag = dict([[v, k] for k, v in load_obj("../data/tag_to_index").items()])
index_to_token = dict([[v, k] for k, v in load_obj("../data/word_to_index").items()])
print("\n example data/labels: \n")
for i, batch in enumerate(train_iter):
    if i<5:
        print("\nbatch: {}, buckets: {}, bucket key: {}, batch size: {}, example feature shape: {}".format(i, len(batch.data), batch.bucket_key, len(batch.data[0]), batch.data[0][0].shape))
        # print("\nexample feature: \n", batch.data[0][0])
        # print("\nexample label: \n", batch.label[0][0])
        #convert back to words to check order
        indexed_utterance = batch.data[0][0][0,:]
        indexed_label = batch.label[0][0]
        utterance = np.vectorize(index_to_token.get)(indexed_utterance.asnumpy())
        label = np.vectorize(index_to_tag.get)(indexed_label.asnumpy())
        print("utterance: {}, label: {}".format(utterance,label))

train_iter.reset()
###############################################################
# create bucketing module to train on variable sequence lengths
###############################################################

#use GPU optimized sequential RNN cell if possible
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

    input_feature_shape = (config.batch_size, features, seq_len)
    input_label_shape = (config.batch_size, seq_len)

    #data placeholders: we are inputting a sequence of data each time.
    seq_data = mx.symbol.Variable('seq_data')
    seq_label = mx.sym.Variable('seq_label')
    print("\ninput data shape: ", seq_data.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\ninput label shape: ", seq_label.infer_shape(seq_label=input_label_shape)[1][0])

    #initialize weight array (to multiply loss later), and broadcast array to same shape as loss
    weights = mx.sym.BlockGrad(mx.sym.Variable(shape=(1,1, num_labels), init=WeightInit(), name='class_weights'))
    label_weights = mx.sym.BlockGrad(mx.sym.broadcast_to(weights, shape=(config.batch_size, seq_len, num_labels), name='broadcast_class_weights'))
    print("\nweights shape: ", label_weights.infer_shape()[1][0])

    #split input features
    tokens = mx.sym.Reshape(mx.sym.transpose(mx.sym.slice_axis(seq_data, axis=1, begin=0, end=1), axes = (0,2,1)),shape = (0,0))
    char_features = mx.sym.Reshape(mx.sym.transpose(mx.sym.slice_axis(seq_data, axis=1, begin=2, end=features), axes=(0, 2, 1)),shape = (0,1,seq_len,-1))
    pos_tags = mx.sym.one_hot(mx.sym.Reshape(mx.sym.slice_axis(seq_data, axis=1, begin=1, end=2), shape = (0,seq_len)), tag_vocab_size)
    print("\ntoken features shape: ", tokens.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\nchar features shape: ", char_features.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\nonehot postag features shape: ", pos_tags.infer_shape(seq_data=input_feature_shape)[1][0])

    print("""\n\t#########################################
        # CHARACTER LEVEL CONVOLUTIONAL COMPONENT
        #########################################""")

    char_embeddings = mx.sym.Embedding(data=char_features, input_dim=char_vocab_size, output_dim=config.char_vectorsize, name='char_embed')
    print("\nembedded char features shape: ", char_embeddings.infer_shape(seq_data=input_feature_shape)[1][0])

    cnn_outputs = []
    for i, filter_size in enumerate(config.char_filter_list):

        print("\n applying filters of size ", filter_size)
        
        #convolutional layer with a kernel that slides over entire words resulting in a 1d output
        convi = mx.sym.Convolution(data=char_embeddings,
                                   kernel=(1, filter_size, config.char_vectorsize), 
                                   stride = (1,1,1), 
                                   num_filter=config.char_filters,
                                   name ="conv_layer_" +  str(i))
        print("\n\tconvolutional output shape: ", convi.infer_shape(seq_data=input_feature_shape)[1][0])

        #apply activation function
        acti = mx.sym.Activation(data=convi, act_type='tanh')

        #take the max value of the convolution, sliding 1 unit (stride) at a time
        pooli = mx.sym.Pooling(data=acti, pool_type='max', kernel=(1, config.max_token_chars - filter_size + 1, 1), stride=(1, 1, 1))
        print("\n\tpooled features shape: ", pooli.infer_shape(seq_data=input_feature_shape)[1][0])

        pooli = mx.sym.transpose(mx.sym.Reshape(pooli, shape = (0,0,0)), axes = (0,2,1))
        print("\n\treshaped/transposed pooled features shape: ", pooli.infer_shape(seq_data=input_feature_shape)[1][0])

        cnn_outputs.append(pooli)

    #combine features from all filters
    cnn_char_features = mx.sym.Concat(*cnn_outputs, dim=2)
    print("\ncnn char features shape: ", cnn_char_features.infer_shape(seq_data=input_feature_shape)[1][0])

    #apply dropout to this layer
    cnn_char_features = mx.sym.Dropout(data=cnn_char_features, p=config.cnn_dropout, mode='training')

    print("""\n\t#########################
        # WORD EMBEDDING FEATURES
        #########################""")

    #create an embedding layer
    word_embeddings = mx.sym.Embedding(data=tokens, input_dim=vocab_size, output_dim=config.word_embedding_vector_length, name='vocab_embed')
    print("\nembedding layer shape: ", word_embeddings.infer_shape(seq_data=input_feature_shape)[1][0])

    #combining all features
    rnn_features = mx.sym.Concat(*[word_embeddings], dim = 2)#, cnn_char_features, pos_tags, word_embeddings
    print("\nall features  shape: ", rnn_features.infer_shape(seq_data=input_feature_shape)[1][0])

    print("""\n\t##############################
        # BIDIRECTIONAL LSTM COMPONENT
        ##############################""")

    #unroll the lstm cell in time, merging outputs
    bi_cell.reset()
    output, states = bi_cell.unroll(length=seq_len, inputs=rnn_features, merge_outputs=True)
    print("\noutputs from all lstm cells in final layer: ", output.infer_shape(seq_data=input_feature_shape)[1][0])

    #reshape outputs so each lstm state size can be mapped to n labels
    output = mx.sym.Reshape(output, shape=(-1,config.lstm_state_size*2), name='r_output')
    print("\nreshaped output shape: ", output.infer_shape(seq_data=input_feature_shape)[1][0])

    #map each output to num labels
    fc = mx.sym.FullyConnected(output, num_hidden=num_labels, name='fc_layer')
    print("\nfully connected layer shape: ", fc.infer_shape(seq_data=input_feature_shape)[1][0])

    #reshape back to same shape as loss will be
    reshaped_fc = mx.sym.reshape(fc, shape = (config.batch_size, seq_len, num_labels))
    print("\nreshaped fc for loss: ", reshaped_fc.infer_shape(seq_data=input_feature_shape)[1][0])

    print("""\n\t#################################
        # WEIGHTED SOFTMAX LOSS COMPONENT
        #################################""")

    #apply softmax function to ensure outputs from fc are between 0 and 1
    sm = mx.sym.softmax(data=reshaped_fc, axis=2, name='softmax_pred')
    print("\nshape after applying softmax to data: ", sm.infer_shape(seq_data=input_feature_shape)[1][0])

    #create a symbol to use with evaluation metrics, since we use a custom loss function
    softmax_output = mx.sym.BlockGrad(data = sm, name = 'softmax')

    #one hot encode label input
    one_hot_labels = mx.sym.one_hot(indices=seq_label, depth=num_labels, name='one_hot_labels')
    print("\nonehot label shape: ", one_hot_labels.infer_shape(seq_label=input_label_shape)[1][0])

    #compute cross entropy loss between predictions and labels
    loss = -((one_hot_labels * mx.sym.log(sm)) + ((1 - one_hot_labels) * mx.sym.log(1 - sm)))
    print("\ncross entropy loss shape: ", loss.infer_shape(seq_data = input_feature_shape, seq_label=input_label_shape)[1][0])

    #symbol to compute the gradient of the loss with respect to the input data
    loss_grad = mx.sym.MakeLoss(loss * label_weights, name='loss_gradient')
    print("\nloss grad shape: ", loss_grad.infer_shape(seq_data=input_feature_shape, seq_label=input_label_shape)[1][0])

    #finally create a symbol group consisting of both the model output (gradient of loss with respect to data) and the softmax layer output (model predictions)
    network = mx.sym.Group([loss_grad, softmax_output])

    return network, ('seq_data',), ('seq_label',)


#####################################
# create a trainable bucketing module
#####################################

model = mx.mod.BucketingModule(sym_gen= sym_gen, 
                               default_bucket_key=train_iter.default_bucket_key, 
                               context = config.context)

# allocate memory to module
model.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)
model.init_params(initializer=mx.init.Uniform(scale=.1))
model.init_optimizer(optimizer=config.optimizer, optimizer_params=config.optimizer_params)

#define a custom metric, which takes the output from an internal layer and calculates precision, recall and f1 score for the entity class
metric = composite_classifier_metrics()

################################################
# fit the model to the training data and save it
################################################

# train x epochs, i.e. going over the data iter one pass
try:
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

        metric.reset()

        for batch in val_iter:
            bucket = batch.bucket_key
            model.forward(batch, is_train=False)       # compute predictions
            model.update_metric(metric, batch.label)   # accumulate metric scores
        print('Epoch %d, Validation %s' % (epoch, metric.get()))

except KeyboardInterrupt:
    print('\n' * 5, '-' * 89)
    print('Exiting from training early, saving model...')

    model.save_params('../results/ner_model')
    print('\n' * 5, '-' * 89)

model.save_params('../results/ner_model')
