
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

import util

class DQN(object):

    def __init__(self, input_config=None):
        """"""
        self.input_config = input_config

        self.shape_layers = input_config.shape_layers
        self.learning_rate = input_config.learning_rate

        self.batch_size = input_config.batch_size
        self.max_episodes = input_config.max_episodes
        self.max_steps_per_episode = input_config.max_steps_per_episode
        # self.output_graph = False
        # 
        self.gamma = input_config.reward_decay # 奖励衰减
        # self.replace_target_iter = input_config.replace_target_iter # 模型更新频率
        self.update_freq = input_config.update_freq
        # self.memory_size = input_config.memory_size # 训练集大小
        self.epsilon_min = input_config.epsilon_greedy # 
        self.epsilon_decrement = input_config.epsilon_greedy_decrement

        self.flag_static_memory = input_config.flag_static_memory

        self.init()
    
    def init(self):
        self.num_episode = 0
        self.num_step = 0
        self.replay_size = 2000
        if self.flag_static_memory:
            DQN.replay_queue = deque(maxlen=self.replay_size)
            DQN.eval_model = self.build_mlp(self.shape_layers)
            DQN.target_model = self.build_mlp(self.shape_layers)
        else:
            self.replay_queue = deque(maxlen=self.replay_size)
            self.eval_model = self.build_mlp(self.shape_layers)
            self.target_model = self.build_mlp(self.shape_layers)

        self.epsilon = 1 if self.epsilon_decrement is not None else self.epsilon_min

        self.score_list = []
        self.history = []
    
    def build_mlp(self, layers_size, is_flatten=False):

        assert len(layers_size) > 1, ('Can not produce network with layer less than 2.')
        print('layers_size : ', layers_size)
        print('last layer', layers_size[-1])
        model = models.Sequential()
        if is_flatten:
            model.add(layers.Flatten())
        if len(layers_size) > 2:
            for i in range(1, len(layers_size) - 1):
                model.add(layers.Dense(layers_size[i], activation='relu'))

        model.add(layers.Dense(layers_size[-1], activation='linear'))
        
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(self.learning_rate),
                      metrics=['mse'])

        return model
    
    def select_action(self, s):
        """预测动作"""
        # 刚开始时，加一点随机成分，产生更多的状态
        if np.random.uniform() < self.epsilon - self.num_step * self.epsilon_decrement:
            self.selected_action = np.random.choice(self.shape_layers[-1])
        else:
            q_values = self.eval_model.predict(np.array([s]))[0]
            print('q_values', str(q_values))
            self.selected_action = np.argmax(q_values)
            print('selected action', self.selected_action)
        return self.selected_action

    def save_model(self, file_path='model_saved.h5'):
        print('model saved')
        self.eval_model.save(file_path)

    def remember(self, s, a, next_s, reward):
        # """car这个场景下, 加快收敛速度"""
        # if next_s[0] >= 0.4:
        #     reward += 1
        # 顺序 state, action, next_state, reward
        self.replay_queue.append((s, a, next_s, reward))

    def train(self):
        if len(self.replay_queue) < self.replay_size:
            return
        self.num_step += 1
        # 每update_freq步，将model的权重赋值给target_model
        if self.num_step % self.update_freq == 0:
            self.target_model.set_weights(self.eval_model.get_weights())

        replay_batch = random.sample(self.replay_queue, self.batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.eval_model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - self.learning_rate) * Q[i][a] + self.learning_rate * (reward + self.gamma * np.amax(Q_next[i]))

        # 传入网络进行训练
        history = self.model.fit(s_batch, Q, verbose=0)
        print(history)
        self.history.append(history.history['mse'])
    
    def step_reset(self, init_state):
        self.current_state = init_state
        self.score = 0

    def step(self, next_s, reward, done, info):
        # self.selected_action = self.select_action(self.current_state)
        # # next_s, reward, done, _ = env.step(a)
        self.remember(self.current_state, self.selected_action, next_s, reward)
        self.train()
        self.score += reward
        self.current_state = next_s
        if done:
            self.score_list.append(self.score)
            print('episode:', self.num_episode, 'score:', self.score, 'max:', max(self.score_list))
            self.num_episode += 1
            return True
        return False
    
    def opponent_turn_update(self, state, reward, done=False):
        self.current_state = state
