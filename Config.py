

## Algorithm section

INITIAL_RUN_NUMBER = None
INITIAL_MODEL_VERSION = None
INITIAL_MEMORY_VERSION = None

absolute_path = None

def initialise_config():
    import os
    global absolute_path
    absolute_path = os.getcwd()

def update_from_dict(obj, kwargs):
    assert type(kwargs) == dict, ('kwargs is not a dictionary.')
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)

class InputConfig(object):

    def __init__(self, **kwargs):
        pass
    
    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value
    
    def __contains__(self, key):
        return key in self._values


    # 官网教程 keras
    # 图像分类
    # optimizer adam
    # loss tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # metrics=['accuracy']
    # 文本分类
    # optimizer adam
    # loss binary_crossentropy
    # metrics=['accuracy']
    # TF Hub 文本分类
    # optimizer adam
    # loss tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # metrics=['accuracy']
    # 回归
    # tf.keras.optimizers.RMSprop(0.001)
    # loss mse
    # metrics=['mae', 'mse']
    # mse MeanSquaredError
    # 保存和加载
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    # 使用 Keras Tuner 调整超参数
    # optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate)
    # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # optimizers.Adam() Adaptive Moment Estimation
    # self.optimizer = optimizers.Adam()
    # optimizers.SGD(learning_rate) Stochastic Gradient Descent, SGD
    # self.optimizer = optimizers.SGD(learning_rate=self.learning_rate)


class InputConfig_Method(InputConfig):

    # loss 可以直接在 compile 里用tf.keras.losses对象
    network_types = ['mlp', 'user_given']
    optimizer_type = ['adam']
    loss_type = ['binary_crossentropy', 'mse', 'sparse_categorical_crossentropy']


    def __init__(self, **kwargs):
        # input params
        ## common
        self.env_name = 'TicTacToe'
        self.learning_rate = 0.001 # 1
        self.num_epochs = 10
        self.batch_size = 32
        self.max_episodes = 2000
        self.max_steps_per_episode = 1000
        self.network_type = 'mlp'
        self.flag_static_memory = False # 是否整个类使用公共的网络，以便做self-play
        self.weighted_replay_queue = False # 是否使用两个队列，分别记录奖励为零和不为零的条目
        ## agent params
        self.shape_layers = None
        self.shape_action = None
        self.shape_feature = None
        ## DQN
        self.reward_decay = 0.9 # gamma
        self.memory_size = 2000 # 500
        self.epsilon_greedy = 0.3
        self.epsilon_greedy_decrement = 0
        self.output_graph = False
        self.update_freq = 200
        ## MLP
        self.layers_size = None # (9,6,1) for 9 inputs, 6 hiddens, 1 outputs
        self.network = None
        self.is_flatten = False
        self.loss = None
        self.optimizer = None
        ## CNN
        self.function_build_cnn = None
        ## MCTS
        self.c_puct = 1

        # update the default params according to kwargs
        update_from_dict(self, kwargs)
        
        # assert self._values['env_name'] in InputConfig.env_type, ('This environment is not supported. Please develop the related code.')
        
        # if self._values['env_name'] == 'TicTacToe':
        #     self._values['action_size'] = self._values['len_map']



class InputConfig_Env(InputConfig):

    env_type = ['TicTacToe']
    
    def __init__(self, **kwargs):
        self._values = {
            # env needed params
            'flag_self_play_view': False,
            
            # env_core needed params
            'size': 3, # if all dimensions have the same size, this parameter could be used. or dimensions could be setted by dims
            'num_dim': 2,
            'dims': None, # shape
            'num_in_a_row': 3, # win_condition, win_length
            'num_players': 2,
            'flag_compute_used_left': True, # 统计 有子位置和空余位置
            # no need to set
            'len_map': 0, # will be automatically compute according dims
            
            # gravity needed params
            'gravity_mode': 'no_gravity', # other possible mode, see modes in GravitySetting
            # mode along_axis
            'gravity_along_axis': 0,
            'gravity_direction_along_axis': 1, # 1 or -1
        }

        if len(kwargs) > 0:
            self._values.update(kwargs)

        if self._values['dims'] == None:
            self._values['dims'] = [self._values['size']] * self._values['num_dim']
        else:
            self._values['num_dim'] = len(self._values['dims'])
        
        self._values['len_map'] = 1
        for i in self._values['dims']:
            self._values['len_map'] *= i
        