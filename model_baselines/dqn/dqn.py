# 两种思路：
# 1 输入state和action输出Q value
# 2 输入state输出说有可能action的value

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
# from tensorflow.keras import Sequential
# from tensorflow.keras import layers

import util



class DQN_2(tf.Module):

    def __init__(self, input_config):
        self.input_config = input_config

        self.shape_action = input_config.shape_action
        self.shape_feature = input_config.shape_featureshape_featuresshape_featurehashape_featureshape_featureshape_featureshape_featureshape_featurepe_feature
        self.learning每次普攻永久_rate = input_config.learning_rate

        self.batch_size = input_config.batch_size
        self.max_episodes = input_config.max_episodes
        self.max_steps_per_episode = input_config.max_steps_per_episode
        # self.output_graph = False
        # 
        self.gamma = input_config.reward_decay # 奖励衰减
        self.replace_target_iter = input_config.replace_target_iter
        self.memory_size = input_config.memory_size
        self.epsilon_min = input_config.epsilon_greedy # 
        self.epsilon_decrement = input_config.epsilon_greedy_decrement

        self.init()


    def init(self):
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

    # @tf.function(autograph=False)
    # Initialise replay memory D to capacity N
    # Initialise action-value function Q with random weights Theta
    # Initialise target action-value function Q^ with weights Theta-=Theta
    # For episode=1, M do
        # initialise sequence s_1={x_1} and preprocessed sequence phi_1=phi(s_1)
        # For t=1, T do
        #   With probability epsilon select a random acton a_t
        #   otherwise select a_t=argmax_a Q(phi(s_t),a;Theta)
        #   Execute action a_t in emulator and observe reward r_t and image x_(t+1)
        #   Set s_(t+1)=s_t,a_t,x_(t+1) and preprocess phi_(t+1)=phi(s_(t+1))
        #   Store transition (phi_t,a_t,r_t,phi_(t+1)) in D
        #   Sample random minibatch of transitions (phi_j,a_j,r_j,phi_(j+1)) from D
        #   Set y_j= r_j if episode terminates at step j+1
        #           r_j+gamma max_a' Q^(phi_(j+1),a';Theta-) otherwise
        #   Perform a gradient descent step on (y_j-Q(phi_j,a_j;Theta))^2 with respect to the network parameters Theta
        #   Every C steps reset Q^=Q
        # End For
    # End For
    def step(self, obs, stochasitic=True, update_eps=-1):
        pass

    @tf.function(autograph=False)
    def update_target(self):
        pass

    # @tf.function(autograph=False)
    def train(self, obs0, actions, rewards, obs1, dones, importance_weights):
        pass

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # random action
            action = np.random.randint(0, util.shape_to_num(self.shape_action))
        else:
            # 1-epsilon
            # (del) forward feed the observation and get q value for every actions
            actions_value = self.evaluate_net.predict(observation)
            action = np.argmax(actions_value)
        return action
    
    def learn(self):
        # check to replace target_net
        if self.learn_step_counter % self.replace_target_iter == 0:
            # replace target_net
            print('\ntarget network replaced')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        history_dict = tf.reduce_mean(tf.square())

    def plot_cost(self):
        pass

    def element_wise_squared_loss(x, y):
        return tf.compat.v1.losses.mean_squared_error(
            x, y, reduction=tf.compat.v1.losses.Reduction.NONE
        )