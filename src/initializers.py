#modules
import mxnet as mx

#custom modules
import config
from misc_modules import load_obj

#which index of the prediction array corresponds to not entity tag?
not_entity_index = load_obj("../data/tag_index_dict")["O"]

# Create and register a custom initializer that initializes weights to 0.1 and biases to 1.
@mx.init.register
class WeightInit(mx.init.Initializer):
    def __init__(self):
        super(WeightInit, self).__init__()
    def _init_weight(self, _, arr):
        #set all weights to 1
        arr[:] = config.entity_weight
        #scale the weight for not entity values
        arr[:, :, not_entity_index] = 1
