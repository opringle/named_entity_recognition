#modules
import mxnet as mx

#preprocessing
split = [0.8, 0.2]

#training
max_training_examples = None
max_val_examples = None

context = mx.cpu() #train on gpu or cpu
buckets =[6,7,8,9,10] #leaving this empty lets MXNet choose best bucket sizes from data

word_embedding_vector_length = 60  #the length of the vector for each unique word in the corpus
lstm_layers = 1 #number of bidirectional lstm layers
lstm_state_size = 27 #choose the number of neurons in each unrolled lstm state
lstm_dropout = 0.0 #dropout applied after each lstm layer

batch_size = 63  # number of training examples to compute gradient with
num_epoch = 12 #number of  times to backprop and update weights
optimizer = 'Adam' #choose algorith for initializing and updating weights
optimizer_params = {"learning_rate": 0.00105, "beta1": 0.9, "beta2": 0.999, "wd": 0.002}

#the loss for missing an entity tag is x times the loss for missing a "not entity" tag
#the loss for incorrectly predicting an entity tag is x times the loss for incorrectly predicting a not entity tag 
entity_weight = 7.5

# bucketing no longer works with weight array
# dropout has no effect
# effect of entity weighting in loss is not as expected
# l2 regularizing prevents loss from going unstable
# in paper the f1 score with no features or cnn component is 80, if I can achieve this I may not need their loss function
