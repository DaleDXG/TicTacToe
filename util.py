
# import Config
import logging
# import tensorflow as tf
# import pickle
import numpy as np
import pandas as pd
# import copy
from collections.abc import Iterable
# import sys
# import os

import trueskill
import itertools
import math

import matplotlib.pyplot as plt

run_folder = '' # os.path.dirname(sys.path[0]) # sys.path[0]

# from IPython import display
# from matplotlib import pyplot as plt
# import d2l


## common part

def add_coordinates(coordinates_1, coordinates_2):
    # could check DimentionMatchingError
    result = []
    for i in range(len(coordinates_1)):
        result.append(coordinates_1[i] + coordinates_2[i])
    return tuple(result)

def multiply_coordinates(coordinates, multiplier):
    result = []
    for i in range(len(coordinates)):
        result.append(coordinates[i] * multiplier)
    return tuple(result)

def add_element_to_tuple(set, new_element, index):
    result = []
    for i in range(len(set)):
        if i == index:
            result.append(new_element)
        result.append(set[i])
    return tuple(result)

def copy_list(list):
    result = []
    for item in list:
        result.append(item)
    return result

def update_from_dict(obj, kwargs):
    assert type(kwargs) == dict, ('kwargs is not a dictionary.')
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)

def shape_to_num(shape):
    result = 1
    if type(shape) == int or type(shape) == np.int64:
        return shape
    else:
        for i in shape:
            result *= shape_to_num(i)
        return result
    # try:
    #     for i in shape:
    #         result *= i
    # except TypeError:
    #     # print(type(shape))
    #     if type(shape) == int:
    #         result = shape
    #     else:
    #         raise TypeError

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, Iterable):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

# 绘图 plot

colour_set = ['green', 'blue', 'yellow', 'red', 'purple']

def plot(muti_data):
    for i, data in enumerate(muti_data):
        plt.plot(data, color=colour_set[i])
    plt.show()

## logging part

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


logger_env = setup_logger('logger_env', run_folder + 'logs/logger_env.log')
logger_trueskill = setup_logger('logger_trueskill', run_folder + 'logs/logger_trueskill.log')
# logger_env.disabled = False

def read_log_trueskill(file_path):
    df = pd.read_csv(file_path, delimiter='\n')


# TrueSkill

def win_probability(team1, team2):
    BETA = trueskill.global_env().beta
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (BETA * BETA) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)


# method part

def load_data(data, batch_size, resize=None):
    pass


### d2l copy

# size = lambda a: tf.size(a).numpy()

# def load_data_fashion_mnist(batch_size, resize=None):
#     """Download the Fashion-MNIST dataset and then load it into memory."""
#     mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
#     # Divide all numbers by 255 so that all pixel values are between
#     # 0 and 1, add a batch dimension at the last. And cast label to int32
#     process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
#                             tf.cast(y, dtype='int32'))
#     resize_fn = lambda X, y: (tf.image.resize_with_pad(X, resize, resize)
#                               if resize else X, y)
#     return (tf.data.Dataset.from_tensor_slices(
#         process(*mnist_train)).batch(batch_size).shuffle(len(
#             mnist_train[0])).map(resize_fn),
#             tf.data.Dataset.from_tensor_slices(
#                 process(*mnist_test)).batch(batch_size).map(resize_fn))

# # Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
# class Accumulator:
#     """For accumulating sums over `n` variables."""
#     def __init__(self, n):
#         self.data = [0.0] * n

#     def add(self, *args):
#         self.data = [a + float(b) for a, b in zip(self.data, args)]

#     def reset(self):
#         self.data = [0.0] * len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# # Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
# class Animator:
#     """For plotting data in animation."""
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
#                  ylim=None, xscale='linear', yscale='linear',
#                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
#                  figsize=(3.5, 2.5)):
#         # Incrementally plot multiple lines
#         if legend is None:
#             legend = []
#         display.set_matplotlib_formats('svg')
#         self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1:
#             self.axes = [self.axes,]
#         # Use a lambda function to capture arguments
#         self.config_axes = lambda: set_axes(self.axes[
#             0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts

#     def add(self, x, y):
#         # Add multiple data points into the figure
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)
#         if not hasattr(x, "__len__"):
#             x = [x] * n
#         if not self.X:
#             self.X = [[] for _ in range(n)]
#         if not self.Y:
#             self.Y = [[] for _ in range(n)]
#         for i, (a, b) in enumerate(zip(x, y)):
#             if a is not None and b is not None:
#                 self.X[i].append(a)
#                 self.Y[i].append(b)
#         self.axes[0].cla()
#         for x, y, fmt in zip(self.X, self.Y, self.fmts):
#             self.axes[0].plot(x, y, fmt)
#         self.config_axes()
#         display.display(self.fig)
#         display.clear_output(wait=True)

# # Defined in file: ./chapter_preliminaries/calculus.md
# def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
#     """Set the axes for matplotlib."""
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.set_xscale(xscale)
#     axes.set_yscale(yscale)
#     axes.set_xlim(xlim)
#     axes.set_ylim(ylim)
#     if legend:
#         axes.legend(legend)
#     axes.grid()

# # Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
# def accuracy(y_hat, y):
#     """Compute the number of correct predictions."""
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = tf.argmax(y_hat, axis=1)
#     cmp = tf.cast(y_hat, y.dtype) == y
#     return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))

# # Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
# def evaluate_accuracy(net, data_iter):
#     """Compute the accuracy for a model on a dataset."""
#     metric = Accumulator(2)  # No. of correct predictions, no. of predictions
#     for X, y in data_iter:
#         metric.add(accuracy(net(X), y), size(y))
#     return metric[0] / metric[1]

# # Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
# def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
#     """Train a model (defined in Chapter 3)."""
#     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
#                         legend=['train loss', 'train acc', 'test acc'])
#     for epoch in range(num_epochs):
#         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
#         test_acc = evaluate_accuracy(net, test_iter)
#         animator.add(epoch + 1, train_metrics + (test_acc,))
#     train_loss, train_acc = train_metrics
#     assert train_loss < 0.5, train_loss
#     assert train_acc <= 1 and train_acc > 0.7, train_acc
#     assert test_acc <= 1 and test_acc > 0.7, test_acc

# # Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
# def train_epoch_ch3(net, train_iter, loss, updater):
#     """The training loop defined in Chapter 3."""
#     # Sum of training loss, sum of training accuracy, no. of examples
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         # Compute gradients and update parameters
#         with tf.GradientTape() as tape:
#             y_hat = net(X)
#             # Keras implementations for loss takes (labels, predictions)
#             # instead of (predictions, labels) that users might implement
#             # in this book, e.g. `cross_entropy` that we implemented above
#             if isinstance(loss, tf.keras.losses.Loss):
#                 l = loss(y, y_hat)
#             else:
#                 l = loss(y_hat, y)
#         if isinstance(updater, tf.keras.optimizers.Optimizer):
#             params = net.trainable_variables
#             grads = tape.gradient(l, params)
#             updater.apply_gradients(zip(grads, params))
#         else:
#             updater(X.shape[0], tape.gradient(l, updater.params))
#         # Keras loss by default returns the average loss in a batch
#         l_sum = l * float(tf.size(y)) if isinstance(
#             loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
#         metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
#     # Return training loss and training accuracy
#     return metric[0] / metric[2], metric[1] / metric[2]
