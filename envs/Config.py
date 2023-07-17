


class InputConfig(object):
    
    def __init__(self, **kwargs):
        self._values = {
            # env needed params
            # env_core needed params
            'size': 3, # if all dimensions have the same size, this parameter
            'num_dim': 2,
            'dims': None, # shape
            'num_in_a_row': 3, # win_condition, win_length
            'len_map': 0,
            'num_players': 2,
            'flag_compute_used_left': False,
            
            # gravity needed params
            'gravity_mode': 'along_axis', # other possible mode, see modes in GravitySetting
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