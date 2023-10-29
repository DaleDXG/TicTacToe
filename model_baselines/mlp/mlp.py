import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import losses
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
# from util import update_from_dict


    
def build_mlp(layers_size, is_flatten=False):

    assert len(layers_size) > 1, ('Can not produce network with layer less than 2.')

    network = Sequential()
    if is_flatten:
        network.add(layers.Flatten())
    if len(layers_size) > 2:
        for i in range(1, len(layers_size) - 1):
            network.add(layers.Dense(layers_size[i], activation='relu'))
    network.add(layers.Dense(layers_size[-1]))

    return network

# change from tf.Module Model
class MLP(Model):
    """
    Member variable:
    num_epochs
    batch_size
    layers_size
    is_flatten

    network
    loss
    optimizerInteger or None. Number of samples per batch of computation. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of a dataset, generators, or keras.utils.Sequence instances (since they generate batches).
    """

    def __init__(self, input_config): # layers_size=None, learning_rate=0.1, num_epochs=10, batch_size=32, is_flatten=False
        super(MLP, self).__init__()
        # self.learning_rate = learning_rate
        self.num_epochs = input_config.num_epochs
        self.batch_size = input_config.batch_size

        assert input_config.layers_size!=None and len(input_config.layers_size) > 1, ('Can not produce network with layer less than 2.')
        self.layers_size = input_config.layers_size

        self.is_flatten = input_config.is_flatten

        if input_config.network != None:
            self.network = input_config.network
        else:
            self.network = build_mlp(self.layers_size, self.is_flatten)

        self.loss = input_config.loss
        self.optimizer = input_config.optimizer
        self.metrics_overwrite = ['accuracy']
        self.compile()

        # self.loss = losses.BinaryCrossentropy
        # self.loss = losses.CategoricalCrossentropy
        # self.loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # optimizers.Adam() Adaptive Moment Estimation
        # self.optimizer = optimizers.Adam()
        # optimizers.SGD(learning_rate) Stochastic Gradient Descent, SGD
        # self.optimizer = optimizers.SGD(learning_rate=self.learning_rate)


    def set_loss(self, loss):
        self.loss = loss
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def compile(self):
        self.network.compile(optimizer=self.optimizer,
                             loss=self.loss,
                             metrics=self.metrics_overwrite)


    # Model.fit is a high-level endpoint that manages its own `tf.function`.
    # Please move the call to `Model.fit` outside of all enclosing `tf.function`s.
    # @tf.function
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):
        if epochs == None:
            epochs = self.num_epochs
        if batch_size == None:
            batch_size = self.batch_size
        if batch_size == -1:
            batch_size = len(y_train)
        # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

        # self.network.compile(optimizer='adam',
        #                     loss='sparse_categorical_crossentropy',
        #                     metrics=['accuracy'])

        if X_val is None or y_val is None:
            self.history = self.network.fit(X_train,
                                            y_train,
                                            epochs=epochs,
                                            batch_size=batch_size)
        else:
            self.history = self.network.fit(X_train,
                                            y_train,
                                            validation_data=(X_val, y_val),
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            verbose=1)
        return self.history.history

        # for epoch in range(epochs):
        #     for X_batch, y_batch in train_dataset:
        #         with tf.GradientTape() as tape:
        #             logits = self.network(X_batch, training=True)
        #             loss_value = self.loss(y_batch, logits)
        #         grads = tape.gradient(loss_value, self.network.trainable_variables)
        #         self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        

    def predict(self, X_test):
        return self.network.predict(X_test)


    def evaluate(self, X, y):
        loss, accuracy = self.network.evaluate(X, y, verbose=2)
        print('\nTest accuracy: %f\nTest loss: %f' % (accuracy, loss))
        return loss, accuracy


    def plot(self):

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # loss
        plt.plot(epochs,
                 loss,
                 'bo',
                 label='Training loss')
        plt.plot(epochs,
                 val_loss,
                 'b',
                 label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        plt.clf() # 清除数字

        # acc
        plt.plot(epochs,
                 acc,
                 'bo',
                 label='Training acc')
        plt.plot(epochs,
                 val_acc,
                 'b',
                 label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()