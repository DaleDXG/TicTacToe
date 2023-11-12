import tensorflow as tf
import numpy as np

# 生成一段假数据，100是生成的数量
# create data
x_data = np.randowm.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
# 权重 W 接近 0.1，bias 接近 0.3