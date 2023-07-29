# Multilayer Perceptron

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from util import Animator

class MLP():

    def __init__(self, input_config):
        self.batch_size = input_config.batch_size
        self.learning_rate = input_config.learning_rate
        self.num_epochs = input_config.num_epochs

        assert len(input.layers_size) > 1, ('Can not produce network with layer less than 2.')
        self.layers_size = input_config.layers_size
        self.net = Sequential()
        self.net.add(layers.Flatten())
        if len(self.layers_size) > 2:
            for i in range(1, len(self.layers_size) - 1):
                self.net.add(layers.Dense(self.layers_size[i], activation='relu'))
        self.net.add(layers.Dense(self.layers_size[-1]))

        self.loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.trainer = optimizers.SGD(learning_rate=self.learning_rate) # Stochastic Gradient Descent, SGD
    
    def train(self, train_iter, test_iter):
        pass
