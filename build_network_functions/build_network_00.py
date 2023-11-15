
import numpy as np
from tensorflow.keras import models, layers, optimizers, losses



def build_cnn_00():
    model = models.Sequential()
    model.add(layers.Conv2D(kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Conv2D(kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Conv2D(kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(9, activation='softmax'))
    return model
    
def build_cnn_01():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=9,
                            kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Conv2D(filters=9,
                            kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Conv2D(filters=1,
                            kernel_size=3,
                            padding='valid'),
                            activation='relu',
                            input_shape=(3,3))
    model.add(layers.Flatten())
    model.add(layers.Dense(9, activation='softmax'))
    return model

def build_cnn_03():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=8,
                            kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Conv2D(filters=8,
                            kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Conv2D(filters=1,
                            kernel_size=3,
                            padding='valid'),
                            activation='relu',
                            input_shape=(3,3))
    model.add(layers.Flatten())
    model.add(layers.Dense(9, activation='softmax'))
    return model

def build_cnn_03():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=8,
                            kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Conv2D(filters=8,
                            kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(3,3)))
    model.add(layers.Flatten())
    # input 8 * 3 * 3 = 72
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(9, activation='softmax'))
    return model
