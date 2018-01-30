#modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import chain

#custom modules
from misc_modules import save_obj
import config

#read in csv of NER training data
df = pd.read_csv("../data/ner_dataset.csv", encoding="ISO-8859-1")

#rename columns because I have OCD
df = df.rename(columns = {"Sentence #" : "utterance_id",
                            "Word" : "token", 
                            "POS" : "POS_tag", 
                            "Tag" : "BILOU_tag"})

#clean utterance_id column
df.loc[:, "utterance_id"] = df["utterance_id"].str.replace('Sentence: ', '')

#fill np.nan utterance ID's with the last valid entry
df = df.fillna(method='ffill')
df.loc[:, "utterance_id"] = df["utterance_id"].apply(int)

#melt BILOU tags and tokens into an array per utterance
df1 = df.groupby("utterance_id")["BILOU_tag"].apply(lambda x: np.array(x)).to_frame().reset_index()
df2 = df.groupby("utterance_id")["token"].apply(lambda x: np.array(x)).to_frame().reset_index()
df3 = df.groupby("utterance_id")["POS_tag"].apply(lambda x: np.array(x)).to_frame().reset_index()

#join the results on utterance id
df = df1.merge(df2.merge(df3, how = "left", on = "utterance_id"), how = "left", on = "utterance_id")

def pad(char_array):
  """pad/slice token array to a fixed length"""
  pad = config.max_token_chars-len(char_array)
  #pad if too short
  if pad>0:
    char_array = np.pad(char_array, pad_width=(0,pad), mode='constant', constant_values=(0, 0))
  #slice if too long
  else:
    char_array = char_array[:config.max_token_chars]
  return char_array

def featurize(x):
  """create a 2d numpy array of features from postags and tokens"""
  token_list = x[0].tolist()
  pos_array = x[1].reshape((1,-1))
  char_array = np.array([pad(np.array(list(token))) for token in token_list]).T
  token_array = x[0].reshape((1,-1))

  feature_array = np.concatenate((token_array, pos_array, char_array), axis=0)

  return feature_array

#get list of feature arrays
x = df.as_matrix(columns = ['token', 'POS_tag']).tolist()

#get list of 2d feature arrays
x = [featurize(row) for row in x]

#get list of tag arrays
y = df['BILOU_tag'].values.tolist()

#make a dictionary from all unique string values
unique_features = set(list(chain.from_iterable([array.flatten().tolist() for array in x])))
feature_to_index = {k:v for v,k in enumerate(unique_features)}


#make a dictionary from all unique entity tags
unique_tags = set(list(chain.from_iterable(y)))
tag_to_index = {k: v for v, k in enumerate(unique_tags)}

#save dicts
save_obj(feature_to_index, "../data/tag_to_index")
save_obj(tag_to_index, "../data/feature_to_index")

def index_array(array,d):
  """map dict to array"""
  if array.ndim == 2:
    for row in list(range(array.shape[0])):
      array[:,:] = np.vectorize(d.get)(array[:,:])
      return array
  else:
    return np.vectorize(d.get)(array)

#use dictionaries to index the arrays
indexed_x = [index_array(array, feature_to_index) for array in x]
indexed_y = [index_array(array, tag_to_index) for array in y]

print(indexed_y[0:1], indexed_x[0:1])

#split into training and test sets
split_index = int(config.split[0] * len(indexed_x))
x_train = indexed_x[:split_index]
x_test = indexed_x[split_index:]
y_train = indexed_y[:split_index]
y_test = indexed_y[split_index:]

#save to file
np.save('../data/x_train.txt', x_train)
np.save('../data/y_train.txt', y_train)
np.save('../data/x_test.txt', x_test)
np.save('../data/y_test.txt', y_test)