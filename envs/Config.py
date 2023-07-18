


class InputConfig(object):
    
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
            'flag_compute_used_left': False,
            'len_map': 0, # no need to set, will be automatically compute according dims
            
            # gravity needed params
            'gravity_mode': 'no_gravity', # other possible mode, see modes in GravitySetting
            'gravity_along_axis': 0
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
    
    
    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value
    
    def __contains__(self, key):
        return key in self._values



if __name__ == '__main__':
    config = InputConfig(dims=(3,4,4))
    print(config['num_dim'])