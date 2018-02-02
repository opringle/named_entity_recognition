#modules
import mxnet as mx

##############
#preprocessing
##############

split = [0.8, 0.2]
max_training_examples = None
max_val_examples = None
max_token_chars = 20

#########
#training
#########

context = mx.gpu() #train on gpu or cpu
buckets =[] #leaving this empty lets MXNet choose best bucket sizes from data

#cnn hyperparams
char_vectorsize = 25
char_filter_list = [3]
char_filters = 100
cnn_dropout = 0.5

#rnn hyperparams
word_embedding_vector_length = 150  #the length of the vector for each unique word in the corpus
lstm_layers = 1 #number of bidirectional lstm layers
lstm_state_size = 275 #choose the number of neurons in each unrolled lstm state
lstm_dropout = 0.6 #dropout applied after each lstm layer

batch_size = 50 # number of training examples to compute gradient with
num_epoch = 80 #number of  times to backprop and update weights
optimizer = 'Adam' #choose algorith for initializing and updating weights
optimizer_params = {"learning_rate": 0.0001, "beta1": 0.9, "beta2": 0.999, "wd": 0.002}

#scale loss for entity labels
entity_weight = 3
