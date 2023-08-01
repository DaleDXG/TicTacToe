

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



class InputConfig_Method(InputConfig):

    def __init__(self, **kwargs):
        # input params
        self.env_name = 'TicTacToe'
        self.learning_rate = 1
        self.num_epochs = 10
        self.batch_size = 32
        self.network_type = 'mlp'
        ## agent params
        self.action_shape = None
        ## MLP
        self.layers_size = None # (9,6,1) for 9 inputs, 6 hiddens, 1 outputs
        self.is_flatten = False
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
            'flag_compute_used_left': True,
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
        