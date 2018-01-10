#modules
import mxnet as mx

#preprocessing
split = [0.8, 0.2]

#training
context = mx.cpu() #train on gpu or cpu
buckets = [5, 10, 15, 20, 30] #leaving this empty lets MXNet choose best bucket sizes from data
batch_size = 1 #number of training examples to compute gradient with 
word_embedding_vector_length = 2 
lstm_state_size = 1
num_epoch = 800 #number of times to backprop and update weights
learning_rate = 0.0105
