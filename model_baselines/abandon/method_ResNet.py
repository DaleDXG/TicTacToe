import tensorflow as tf
import numpy as np



class Residual_CNN_Unit(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same', strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    # forward
    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)

###定义模型
class res_model(tf.nn):
    def __init__(self, reg_const, learning_rate, input_dim, output_dim):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bn1 = tf.nn.batch_normalization()
        self.conv1 = tf.nn.Conv2D() #same输入与输出形状相同
        self.pooling1 = tf.nn.max_pool()

    ###残差网络增加的部分,输入值和卷积输出值进行拼接，x意为输入值，y意为经过卷积输出的值
    def res_add(x,y):
        y = tf.concat((x,y),axis=0)
        return y

    def foward(self,x):
        #检验输入的数据：
        batchsize = x.size()[0]
        print('检验输入的数据形状:', x.shape)

        #构建网络
        filter = [1, 1, 1, 1]  # 1×1的卷积核尺寸，小于等于输入的尺寸
        x = tf.nn.relu(self.bn1(self.conv1(x, filter, strides = [1, 1, 1, 1], padding = 'SAME')))#第一次，卷积-标准化-激活
        #第二次卷积加残差
        y = self.bn1(self.conv1(x)) #卷积-标准化
        y = self.res_add(x, y)
        y = tf.nn.relu(y)  #激活
        y = tf.nn.max_pool(y,ksize=[1, 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool") #最大池化

        return y



if __name__ == '__main__':
    ## 定义输入数据
    input = tf.Variable(tf.random_normal([1, 2, 3], stddev=0.1))
    print("input:", input)
    # x2 = tf.constant([[2],[3]])
    # x = tf.matmul(x1,x2)
    ## 创建权重
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
    biases = tf.Variable(tf.zeros([200]), name="biases")

    # 初始化
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        result = sess.run(init_op)
        print("session运行了x的结果:", result)

    x = tf.placeholder(tf.float32, shape=(1, 2))
    sess.run(y, feed_dict={x: [[0.5, 0.6]]})

    learning_rate = 0.3
    loss = tf.reduce_mean(tf.square(y_ - y))
    ###训练优化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)