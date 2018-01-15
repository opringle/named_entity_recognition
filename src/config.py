#modules
import mxnet as mx

#preprocessing
split = [0.8, 0.2]

#training
max_training_examples = 2000
max_val_examples = 200

context = mx.cpu() #train on gpu or cpu
buckets = [70] #leaving this empty lets MXNet choose best bucket sizes from data

word_embedding_vector_length = 600  #the length of the vector for each unique word in the corpus
lstm_layers = 1
lstm_state_size = 27 #choose the number of neurons in each unrolled lstm state
lstm_dropout = 0.0

batch_size = 10  # number of training examples to compute gradient with
num_epoch = 40 #number of times to backprop and update weights
optimizer = 'sgd' #choose algorith for initializing and updating weights
optimizer_params = {"learning_rate": 0.105, "wd" : 0}

#rescaling the loss just changes the speed that the model learns, not what it learns! do i need to rescale grad isntead?
not_entity_loss_scale = 0.0000000001
entity_loss_scale = 1000
