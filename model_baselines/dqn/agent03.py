
# agent03

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
import math
# import copy
import numpy as np
from tensorflow.keras import models, layers, optimizers, losses

import util

class DQN(object):

    def __init__(self, name='', input_config=None):
        """"""
        self.name = name
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
        self.replay_size = input_config.memory_size # 2000
        self.epsilon_min = input_config.epsilon_greedy # 
        self.epsilon_decrement = input_config.epsilon_greedy_decrement

        if input_config.function_build_cnn != None:
            self.function_build_cnn = input_config.function_build_cnn
        self.flag_static_memory = input_config.flag_static_memory
        self.weighted_replay_queue = input_config.weighted_replay_queue

        self.init()
    
    def init(self):
        self.num_episode = -1
        self.total_step = 0

        self.replay_queue = deque(maxlen=self.replay_size)
        if self.weighted_replay_queue == True:
            self.replay_queue_zero = deque(maxlen=self.replay_size)

        if self.flag_static_memory:
            operated_entity = DQN
        else:
            operated_entity = self
        if self.input_config.network_type == 'mlp':
            operated_entity.eval_model = self.build_mlp(self.shape_layers)
            operated_entity.target_model = self.build_mlp(self.shape_layers)
        elif self.input_config.network_type == 'user_given':
            operated_entity.eval_model = self.function_build_cnn()
            operated_entity.target_model = self.function_build_cnn()

        self.epsilon = 1 if self.epsilon_decrement==0 else self.epsilon_min # is not None

        self.score_list = []
        self.history = []
    
    def build_mlp(self, layers_size, is_flatten=False):

        assert len(layers_size) > 1, ('Can not produce network with layer less than 2.')
        # print('layers_size : ', layers_size)
        # print('last layer', layers_size[-1])
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
        if np.random.uniform() < self.epsilon: # self.epsilon - self.total_step * self.epsilon_decrement:
            self.selected_action = np.random.choice(self.shape_layers[-1])
        else:
            q_values = self.eval_model.predict(np.array([s]))[0]
            # print('q_values', str(q_values))
            self.selected_action = np.argmax(q_values)
            # print('selected action', self.selected_action)
        if self.epsilon_decrement != 0 and self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement
        return self.selected_action

    def save_model(self, file_path='model_saved.h5'):
        # print('model saved')
        self.eval_model.save(file_path)

    def remember(self, s, a, next_s, reward):
        # """car这个场景下, 加快收敛速度"""
        # if next_s[0] >= 0.4:
        #     reward += 1
        # 顺序 state, action, next_state, reward
        if reward == 0:
            self.replay_queue_zero.append((s, a, next_s, reward))
        else:
            self.replay_queue.append((s, a, next_s, reward))

    def train(self):
        # if len(self.replay_queue) < self.replay_size:
        #     return
        self.total_step += 1
        # 每update_freq步，将model的权重赋值给target_model
        if self.total_step % self.update_freq == 0:
            self.target_model.set_weights(self.eval_model.get_weights())

        if self.weighted_replay_queue:
            # actually mean the reward is not zero
            weight_none_zero_sample = 0.8
            num_none_zero_sample = math.ceil(self.replay_size * weight_none_zero_sample)
            num_zero_sample = math.floor(self.replay_size * (1 - weight_none_zero_sample))
            # sample_size = 0
            if len(self.replay_queue) + len(self.replay_queue_zero) < self.replay_size:
                replay_batch = []
                replay_batch.extend(self.replay_queue)
                replay_batch.extend(self.replay_queue_zero)
            elif len(self.replay_queue) < num_none_zero_sample:
                # 非零不够
                replay_batch = random.sample(self.replay_queue_zero, self.replay_size - len(self.replay_queue))
                replay_batch.extend(self.replay_queue)
            elif len(self.replay_queue_zero) < num_zero_sample:
                # 零不够
                replay_batch = random.sample(self.replay_queue, self.replay_size - len(self.replay_queue_zero))
                replay_batch.extend(self.replay_queue_zero)
            else:
                # sample_size = self.replay_size
                # print('sample_size: ', sample_size)
                replay_batch = random.sample(self.replay_queue, num_none_zero_sample)
                replay_batch.extend(random.sample(self.replay_queue_zero, num_zero_sample))
        else:
            sample_size = 0
            if len(self.replay_queue) < self.replay_size:
                sample_size = len(self.replay_queue)
            else:
                sample_size = self.replay_size
            # print('sample_size: ', sample_size)
            replay_batch = random.sample(self.replay_queue, sample_size)

        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])
        # print('s_batch: \n', s_batch)
        # print('next_s_batch: \n', next_s_batch)
        Q = self.eval_model.predict(s_batch)
        # print('Q shape: ', str(np.shape(Q)))
        # print('Q: ', Q)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            # print('i: ', i)
            _, a, _, reward = replay
            # print('a: ', a)
            Q[i][a] = (1 - self.learning_rate) * Q[i][a] + self.learning_rate * (reward + self.gamma * np.amax(Q_next[i]))

        # 传入网络进行训练
        history = self.eval_model.fit(s_batch, Q, verbose=0)
        store_his = history.history['mse'][0]
        util.logger_env.info('history:' + str(store_his))
        self.history.append(history.history['mse'])
    
    def reset(self, init_state):
        self.current_state = init_state
        self.score = 0
        self.num_episode += 1

    def step(self, next_s, reward, done, info):
        # self.selected_action = self.select_action(self.current_state)
        # # next_s, reward, done, _ = env.step(a)
        self.remember(self.current_state, self.selected_action, next_s, reward)
        self.train()
        self.score += reward
        self.current_state = next_s
        if done:
            self.score_list.append(self.score)
            util.logger_env.info('episode:' + str(self.num_episode) + '_score:' + str(self.score) + '_max:' + str(max(self.score_list)))
            return True
        return False
