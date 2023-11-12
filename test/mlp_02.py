
import tensorflow as tf

from model_baselines.mlp.mlp import MLP
import Config
import torch
from tensorflow.keras import losses
import d2l



input_config = Config.InputConfig_Method(num_epochs=10, batch_size=256, learning_rate=0.1)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_test[..., tf.newaxis].astyper('float32')

# 使用tf.data来将数据集切分为batch以及混淆数据集
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



# train_iter, test_iter = torch.load_data_fashion_mnist(256)
mlp = MLP(input_config)
mlp.train(train_iter)