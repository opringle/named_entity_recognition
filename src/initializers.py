#modules
import mxnet as mx

#custom modules
import config
from data_helpers import load_obj

#which index of the prediction array corresponds to not entity tag?
not_entity_index = load_obj("../data/tag_to_index")["O"]

# Create and register a custom initializer that initializes weights
@mx.init.register
class WeightInit(mx.init.Initializer):
    def __init__(self):
        super(WeightInit, self).__init__()
    def _init_weight(self, _, arr):
        #set entity loss weights
        arr[:] = config.entity_weight
        #set not entity loss weights
        arr[:, :, not_entity_index] = 1
