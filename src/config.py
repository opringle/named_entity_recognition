#modules
import mxnet as mx

#preprocessing
split = [0.8, 0.2]

#training
max_training_examples = 100
max_val_examples = 10
max_token_chars = 20

context = mx.cpu() #train on gpu or cpu
buckets =[] #leaving this empty lets MXNet choose best bucket sizes from data

word_embedding_vector_length = 3  #the length of the vector for each unique word in the corpus
lstm_layers = 1 #number of bidirectional lstm layers
lstm_state_size = 1 #choose the number of neurons in each unrolled lstm state
lstm_dropout = 0.0 #dropout applied after each lstm layer

batch_size = 9  # number of training examples to compute gradient with
num_epoch = 12 #number of  times to backprop and update weights
optimizer = 'Adam' #choose algorith for initializing and updating weights
optimizer_params = {"learning_rate": 0.00105, "beta1": 0.9, "beta2": 0.999, "wd": 0.002}

#scale loss for entity labels
entity_weight = 1000
