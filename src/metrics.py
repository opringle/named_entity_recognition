import mxnet as mx
import numpy as np

#custom modules
from misc_modules import load_obj

#read in dictionary mapping BILOU entity tags to integer indices
tag_dict = load_obj("../data/tag_index_dict")
outside_tag_index = tag_dict["O"]

def classifer_metrics(label, pred):
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
    corr_pred = (prediction == label) == (pred_is_entity == True)

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

    return precision, recall, f1

def entity_precision(label, pred):
    return classifer_metrics(label, pred)[0]

def entity_recall(label, pred):
    return classifer_metrics(label, pred)[1]

def entity_f1(label, pred):
    return classifer_metrics(label, pred)[2]

def composite_classifier_metrics():

    metric1 = mx.metric.CustomMetric(feval=entity_precision,
                           name='precision',
                           output_names=['softmax_output'],
                           label_names=['seq_label'])

    metric2 = mx.metric.CustomMetric(feval=entity_recall,
                           name='recall',
                           output_names=['softmax_output'],
                           label_names=['seq_label'])   
    metric3 = mx.metric.CustomMetric(feval=entity_f1,
                           name='f1 score',
                           output_names=['softmax_output'],
                           label_names=['seq_label'])

    metrics = [metric1, metric2, metric3]

    return mx.metric.CompositeEvalMetric(metrics)

def accuracy(label, pred):
    return np.mean(label == np.argmax(pred, 1))
