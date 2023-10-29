
import tensorflow as tf
print('Tensorflow version:', tf.__version__)

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import Module

class MLP_Module(tf.Module):

    def __init__(self, input_config):
        super(MLP_Module, self).__init__()
        self.num_epochs = input_config.num_epochs
        self.batch_size = input_config.batch_size

        assert input_config.layers_size!=None and len(input_config.layers_size) > 1, ('Can not produce network with layer less than 2.')
        self.layers_size = input_config.layers_size

        self.is_flatten = input_config.is_flatten

        if input_config.network != None:
            self.network = input_config.network
        else:
            self.network = self.__build_mlp(self.layers_size, self.is_flatten)

        self.loss = input_config.loss
        self.optimizer = input_config.optimizer
        self.metrics_overwrite = ['accuracy']
    
    def __build_mlp(self):
        pass

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)