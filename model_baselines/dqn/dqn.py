# 两种思路：
# 1 输入state和action输出Q value
# 2 输入state输出说有可能action的value

import tensorflow as tf
from model_baselines.mlp import mlp



class DQN_2(tf.Module):

    def __init__(self, input_config):
        pass

    @tf.function
    def step(self, obs, stochasitic=True, update_eps=-1):
        pass

    @tf.function()
    def train(self, obs0, actions, rewards, obs1, dones, importance_weights):
        pass

    @tf.function(autograph=False)
    def update_target(self):
        pass

