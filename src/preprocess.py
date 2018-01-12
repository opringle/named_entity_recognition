#modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#custom modules
from custom_methods import save_obj
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
df1 = df.groupby("utterance_id")["BILOU_tag"].apply(list).to_frame().reset_index()
df2 = df.groupby("utterance_id")["token"].apply(list).to_frame().reset_index()

#join the results on utterance id
df = df1.merge(df2, how = "left", on = "utterance_id")

# we need to pad sentences to be atleast as long as the min bucket size
min_sentence_length = min(config.buckets)

# pad all other sentences to this length
def pad(x, max_l):

  pads = max_l - len(x)

  if pads > 0:
    padded_sentence = x + [""] * pads
    padded_tags = x + ["O"] * pads

  else:
    padded_sentence = x
    padded_tags = x

  return padded_sentence, padded_tags

df["token"] = df["token"].apply(lambda x: pad(x, min_sentence_length)[0])
df["BILOU_tag"] = df["BILOU_tag"].apply(lambda x: pad(x, min_sentence_length)[1])

print(df.head(3))
print(df.iloc[2,1])
print(df.iloc[2,2])

#get dictionary mapping BILOU tags to indices and save it
unique_tags = list(set([a for b in df.BILOU_tag.tolist() for a in b]))
tag_indices = list(range(len(unique_tags)))
tag_index_dict = dict(zip(unique_tags, tag_indices))
save_obj(tag_index_dict, "../data/tag_index_dict")

#get dictionary mapping unique words to indices and save it
unique_words = list(set([a for b in df.token.tolist() for a in b]))
word_indices = list(range(len(unique_words)))
word_index_dict = dict(zip(unique_words, word_indices))
save_obj(word_index_dict, "../data/word_index_dict")

#index padded_tag lists and padded utterances
df["indexed_tags"] = df["BILOU_tag"].apply(lambda x: [tag_index_dict.get(tag) for tag in x])
df["indexed_utterance"] = df["token"].apply(lambda x: [word_index_dict.get(word) for word in x])

#get a list of list of int for data and labels
data = df.indexed_utterance.values.tolist()
label = df.indexed_tags.values.tolist()

#split into training and test sets
split_index = int(config.split[0] * len(data))
x_train = data[:split_index]
x_test = data[split_index:]
y_train = label[:split_index]
y_test = label[split_index:]

#save to file
file = open('../data/x_train.txt', 'w')
for item in x_train:
  file.write("%s\n" % item)

file = open('../data/x_test.txt', 'w')
for item in x_test:
  file.write("%s\n" % item)

file = open('../data/y_train.txt', 'w')
for item in y_train:
  file.write("%s\n" % item)

file = open('../data/y_test.txt', 'w')
for item in y_test:
  file.write("%s\n" % item)
