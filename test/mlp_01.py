
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from model_baselines.mlp import build_mlp

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 这一步应该是把黑白图像，归一化到 0-1 之间
x_train, x_test = x_train / 255.0, x_test / 255.0

model = build_mlp((784, 128, 10), is_flatten=True)

# 对每个样本，模型都会返回一个包含logits或log-odds分数的向量，每个类一个
# 虽然看不懂，但先抄下来
# gpt：是将 TensorFlow 模型的权重参数（或张量）转换为 NumPy 数组的一种方法
predictions = model(x_train[:1]).numpy()
# tf.nn.softmax 函数将这些logits转换为每个类的概率
tf.nn.softmax(predictions).numpy()
# 注：可以将 tf.nn.softmax 烘焙到网络最后一层的激活函数中。
# 虽然这可以使模型输出更易解释，但不建议使用这种方式，
# 因为在使用 softmax 输出时不可能为所有模型提供精确且数值稳定的损失计算。

# 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
