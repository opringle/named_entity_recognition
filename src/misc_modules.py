import pickle
import numpy as np

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def featurize(token_list):

    #convert the list to an array
    token_array = np.array(token_list)

    #define lambda function to split a token into an array of characters
    f = lambda token: np.array(list(token))

    




