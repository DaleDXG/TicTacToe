
# 两种思路：
# 1 输入state和action输出Q value
# 2 输入state输出说有可能action的value

'''
replay / memory

from collections import deque

init
self.replay_size = 2000
self.replay_queue = deque(maxlen=self.replay_size)

train
'''


from collections import deque
import random
import numpy as np
from tensorflow.keras import models, layers, optimizers, losses

class DQN(object):

    def __init__(self, input_config=None):
        """"""
        self.input_config = input_config

        self.shape_action = input_config.shape_action
        self.shape_feature = input_config.shape_featureshape_featuresshape_featurehashape_featureshape_featureshape_featureshape_featureshape_featurepe_feature
        self.learning_rate = input_config.learning_rate

        self.batch_size = input_config.batch_size
        self.max_episodes = input_config.max_episodes
        self.max_steps_per_episode = input_config.max_steps_per_episode
        # self.output_graph = False
        # 
        self.gamma = input_config.reward_decay # 奖励衰减
        # self.replace_target_iter = input_config.replace_target_iter # 模型更新频率
        self.update_freq = input_config.update_freq
        self.memory_size = input_config.memory_size  # 训练集大小
        self.epsilon_min = input_config.epsilon_greedy # 
        self.epsilon_decrement = input_config.epsilon_greedy_decrement

        self.init()
    
    def init(self):
        self.step = 0
        self.replay_size = 2000
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()

        self.epsilon = 1 if self.epsilon_decrement is not None else self.epsilon_min

        # total learning step
        self.learn_step_counter = 0
        # initialise zero memory [s,a,r,s_]
        self.memory = np.zeros((self.memory_size, util.shape_to_num(self.shape_feature) + 2))
        self.memory_counter = 0

        # consist of [target_net, evaluate_net]
        self.evaluate_net = MLP(self.input_config)
        self.evaluate_net.set_optimizer(optimizers.Adam())
        # (y_true R+gamma*maxQ(s'), y_pred Q(s2))
        self.evaluate_net.set_loss(losses.MSE())
        self.target_net = self.evaluate_net

        self.history = []

    # ab
    def create_model(self):
        """创建一个隐藏层为100的神经网络"""
        STATE_DIM, ACTION_DIM = 2, 3
        model = models.Sequential([
            layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation='linear')
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model
    
    def act(self, s, epsilon=0.1):
        """预测动作"""
        # 刚开始时，加一点随机成分，产生更多的状态
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]))[0])

    def save_model(self, file_path='model_saved.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward):
        """car这个场景下, 加快收敛速度"""
        if next_s[0] >= 0.4:
            reward += 1

        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # 每update_freq步，将model的权重赋值给target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        # 传入网络进行训练
        self.model.fit(s_batch, Q, verbose=0)
