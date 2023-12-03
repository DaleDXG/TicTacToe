# Multilayer Perceptron

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import losses
from tensorflow.keras import optimizers
import util

class MLP(Model):

    def __init__(self, name, input_config, **network_kwargs):
        super(MLP, self).__init__(name=name)
        self.batch_size = input_config.batch_size
        self.learning_rate = input_config.learning_rate
        self.num_epochs = input_config.num_epochs

        assert len(input_config.layers_size) > 1, ('Can not produce network with layer less than 2.')
        self.layers_size = input_config.layers_size
        self.net = Sequential()
        self.net.add(layers.Flatten())
        if len(self.layers_size) > 2:
            for i in range(1, len(self.layers_size) - 1):
                self.net.add(layers.Dense(self.layers_size[i], activation='relu'))
        self.net.add(layers.Dense(self.layers_size[-1]))

        self.loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.trainer = optimizers.SGD(learning_rate=self.learning_rate) # Stochastic Gradient Descent, SGD
    
    def call(self):
        pass
    
    def train(self, train_iter, test_iter):
        animator = util.Animator(xlabel='epoch', xlim=[1, self.num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch(train_iter)
            test_acc = util.evaluate_accuracy(self.net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        # assert train_loss < 0.5, train_loss
        # assert train_acc <= 1 and train_acc > 0.7, train_acc
        # assert test_acc <= 1 and test_acc > 0.7, test_acc

    def train_epoch(self, train_iter):
        # Sum of training loss, sum of training accuracy, no. of example
        metric = util.Accumulator(3)
        for X, y in train_iter:
            # Compute gradients and update parameters
            with tf.GradientTape() as tape:
                y_hat = self.net(X) # forward
                # Keras implementations for loss takes (labels, predictions)
                # instead of (predictions, labels) that users might implement
                # in this book, e.g. `cross_entropy` that we implemented above
                if isinstance(self.loss, losses.Loss):
                    l = self.loss(y, y_hat) # loss
                else:
                    l = self.loss(y_hat, y)
            if isinstance(self.trainer, optimizers.Optimizer):
                params = self.net.trainable_variables
                grads = tape.gradient(l, params) 
                self.trainer.apply_gradients(zip(grads, params))
            else:
                self.trainer(X.shape[0], tape.gradient(l, self.trainer.params))
            # Keras loss by default returns the average loss in a batch
            l_sum = l * float(tf.size(y)) if isinstance(self.loss, losses.Loss) else tf.reduce_sum(l)
            metric.add(l_sum, util.accuracy(y_hat, y), tf.size(y))
        # Return training loss and training accuracy
        return metric[0] / metric[2], metric[1] / metric[2]
