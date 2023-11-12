import Config
from model_baselines.mlp.mlp import MLP

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

input_config = Config.InputConfig_Method(num_epochs=40,
                                         batch_size=512,
                                         layers_size=('Embedding','GlobalAveragePooling1D','16_relu','1_sigmoid'),
                                         # learning_rate=0.1,
                                         optimizer='adam',
                                         loss='binary_crossentropy')
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
input_config.network = model

# 数据
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


mlp = MLP(input_config)

mlp.train(partial_x_train, partial_y_train, x_val, y_val)

loss, accuracy = mlp.evaluate(test_data, test_labels)

mlp.plot()
