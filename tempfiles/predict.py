#########################################
# create a separate module for predicting
#########################################

# #TODO: pred shape is not reliable when batch size not a multiple of number of training examples

# # because we are designing our own loss function, the model output symbol now returns the gradient of the loss with respect to the input data
# # to deal with this we need to create a separate module for prediction, that takes the output from an intermediate symbol

# internal_symbols = model.symbol.get_internals()  # get all internal symbols
# softmax_sym_index = internal_symbols.list_outputs().index('softmax_pred_output')  # find the index of the softmax prediction output layer
# prediction_symbol = internal_symbols[softmax_sym_index] # retrive softmax pred symbol

# # create module from internal symbol
# model_pred = mx.mod.Module(symbol=prediction_symbol,data_names=('seq_data',), label_names=None)

# # allocate memory given the input data and label shapes
# model_pred.bind(data_shapes=train_iter.provide_data, label_shapes = None)

# # initialize parameters by uniform random numbers
# model_pred.init_params(initializer=mx.init.Uniform(scale=.1))

# #set the parameters of the prediction module to the learned values
# model_pred.set_params(arg_params=model.get_params()[0], aux_params=model.get_params()[1]) # pass learned weights to prediction model


######################################################
# ensure sentences are matched with entities correctly
######################################################

# #load in dict mapping indices back to words
# word_to_index = load_obj("../data/word_index_dict")
# index_to_word = dict([(v, k) for k, v in word_to_index.items()])
# tag_to_index = load_obj("../data/tag_index_dict")
# index_to_tag = dict([(v, k) for k, v in tag_to_index.items()])

# train_iter.reset()
# for i, batch in enumerate(train_iter):
#     if i == 1:
#         data = batch.data[0].asnumpy().tolist()
#         labels = batch.label[0].asnumpy().tolist()

#         #map dict to index values to reproduce sentences
#         sentences_train = [[index_to_word[index] for index in utterance if index != -1] for utterance in data]
#         sentences_test = [[index_to_word[index] for index in utterance if index != -1] for utterance in data]
#         tags_train = [[index_to_tag[index] for index in tag_sequence if index != -1] for tag_sequence in labels]
#         tags_test = [[index_to_tag[index] for index in tag_sequence if index != -1] for tag_sequence in labels]

        # for i, sentence in enumerate(sentences_train):
        #     if i < 3:
        #         print("\nsentence>tag mapping: %d" % (i))
        #         for j in list(range(len(sentence))):
        #             print("\ntoken: ", sentences_train[i][j], "\ntag: ", tags_train[i][j])

train_iter.reset()
