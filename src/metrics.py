import mxnet as mx
import numpy as np

#custom modules
from misc_modules import load_obj

#read in dictionary mapping BILOU entity tags to integer indices
tag_dict = load_obj("../data/tag_index_dict")
outside_tag_index = tag_dict["O"]

def entity_F1_score(label, pred):
    """computes the F1 score
    F = 2 * Precision * Recall / (Recall + Precision)"""

    #take highest probability as the prediction of the entity for each word
    prediction = np.argmax(pred, axis=1)
    
    label = label.astype(int)

    #define if the prediction is an entity or not
    not_entity_index = load_obj("../data/tag_index_dict")["O"]
    pred_is_entity = prediction != not_entity_index
    label_is_entity = label != not_entity_index

    #is the prediction correct?
    corr_pred = prediction == label

    #how many entities are there?
    num_entities = np.sum(label_is_entity)
    entity_preds = np.sum(pred_is_entity)

    #how many times did we correctly predict an entity?
    correct_entitites = np.sum(corr_pred[pred_is_entity])

    #precision: when we predict entity, how often are we right?
    precision = correct_entitites/entity_preds
    if entity_preds == 0:
        precision = np.nan

    #recall: of the things that were an entity, how many did we catch?
    recall = correct_entitites / num_entities
    if num_entities == 0:
        recall = np.nan

    #f1 score combines the two 
    f1 = 2 * precision * recall / (precision + recall)

    #print("\nprecision: ", precision, "\nrecall: ", recall)

    return f1

def cust_acc(label, pred):

    label = label.astype(int)
    prediction = np.argmax(pred, axis=1)

    print("\nlabel example: \n", label[0])
    print("\nprediction example: \n", prediction[0])

    return np.mean(prediction == label)

def cust_loss(label, pred):

    nb_classes = pred.shape[1]

    # one hot encode label
    label = np.eye(nb_classes)[label.astype(int)]
    label = label.transpose((0,2,1))

    # print(pred.shape, label.shape)
    # print(pred[0], label[0])

    loss = (label - pred)**2

    return np.sum(loss)


