import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import losses
from tensorflow.keras import optimizers
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

class MLP(tf.Module):

    def __init__(self, layers_size=None, learning_rate=0.1, num_epochs=10, batch_size=32, is_flatten=False):
        super(tf.Module, self).__init__()
        # self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        assert layers_size!=None and len(layers_size) > 1, ('Can not produce network with layer less than 2.')
        self.layers_size = layers_size

        self.is_flatten = is_flatten

        self.network = build_mlp(self.layers_size, self.is_flatten)

        self.loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        # optimizers.Adam() Adaptive Moment Estimation
        self.optimizer = optimizers.Adam()
        # optimizers.SGD(learning_rate) Stochastic Gradient Descent, SGD
        # self.optimizer = optimizers.SGD(learning_rate=self.learning_rate)

    @tf.function
    def train(self, X_train, y_train, epochs=None, batch_size=None):
        if epochs == None:
            epochs = self.num_epochs
        if batch_size == None:
            batch_size = self.batch_size
        if batch_size == -1:
            batch_size = len(y_train)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

        # self.network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # self.network.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        for epoch in range(epochs):
            for X_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    logits = self.network(X_batch, training=True)
                    loss_value = self.loss(y_batch, logits)
                grads = tape.gradient(loss_value, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        

    def predict(self, X_test):
        return self.network.predict(X_test)
