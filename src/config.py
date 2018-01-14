#modules
import mxnet as mx

#preprocessing
split = [0.8, 0.2]

#training
max_training_examples = 500
max_val_examples = 200

context = mx.cpu() #train on gpu or cpu
buckets = [20] #leaving this empty lets MXNet choose best bucket sizes from data

word_embedding_vector_length = 600  #the length of the vector for each unique word in the corpus
lstm_layers = 1
lstm_state_size = 275 #choose the number of neurons in each unrolled lstm state
lstm_dropout = 0.68

batch_size = 9  # number of training examples to compute gradient with
num_epoch = 80 #number of times to backprop and update weights
optimizer = 'sgd' #choose algorith for initializing and updating weights
optimizer_params = {"learning_rate": 0.0105, "wd" : 0}
