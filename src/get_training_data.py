#modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#custom modules
from custom_methods import save_obj

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

#melt BILOU tags and tokenbsinto an array per utterance
df1 = df.groupby("utterance_id")["BILOU_tag"].apply(list).to_frame().reset_index()
df2 = df.groupby("utterance_id")["token"].apply(list).to_frame().reset_index()

#join the results on utterance id
df = df1.merge(df2, how = "left", on = "utterance_id")

#find longest sentence in training data
max_sentence_length = np.max(df.token.apply(len))

#pad all other sentences to this length
def pad(x, max_l):
    pads = max_l - len(x)
    padded_sentence = x + [""] * pads
    padded_tags = x + ["O"] * pads

    return padded_sentence, padded_tags

df["padded_utterance"] = df["token"].apply(lambda x: pad(x, max_sentence_length)[0])
df["padded_tags"] = df["BILOU_tag"].apply(lambda x: pad(x, max_sentence_length)[1])

#select columns we want
df = df[["utterance_id", "padded_utterance", "padded_tags"]]

#get dictionary mapping BILOU tags to indices and save it
unique_tags = list(set([a for b in df.padded_tags.tolist() for a in b]))
tag_indices = list(range(len(unique_tags)))
tag_index_dict = dict(zip(unique_tags, tag_indices))
save_obj(tag_index_dict, "../data/tag_index_dict")


#get dictionary mapping unique words to indices and save it
unique_words = list(set([a for b in df.padded_utterance.tolist() for a in b]))
word_indices = list(range(len(unique_words)))
word_index_dict = dict(zip(unique_words, word_indices))
save_obj(word_index_dict, "../data/word_index_dict")

#index padded_tag lists and padded utterances
df["indexed_padded_tags"] = df["padded_tags"].apply(lambda x: [tag_index_dict.get(tag) for tag in x])
df["indexed_padded_utterance"] = df["padded_utterance"].apply(lambda x: [word_index_dict.get(word) for word in x])

#save features (indexed_padded_utterance) and labels (indexed_padded_tags) as numpy arrays for MXNet
x = pd.DataFrame(df["indexed_padded_utterance"].tolist()).as_matrix()
y = pd.DataFrame(df["indexed_padded_tags"].tolist()).as_matrix()

#randomly select 80% of the rows for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#save numpy arrays for MXNet 
np.save("../data/x_train.npy", x_train)
np.save("../data/x_test.npy", x_test)
np.save("../data/y_train.npy", y_train)
np.save("../data/y_test.npy", y_test)



