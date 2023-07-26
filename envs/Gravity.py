import util



class GravitySetting(object):

    modes = ('no_gravity', 'along_axis', 'toward_axis', 'toward_point')

    def __init__(self, input_config=None):
        assert input_config['gravity_mode'] in GravitySetting.modes, (
            'unknow mode is provided'
        )
        self.mode = input_config['gravity_mode']
        self.shape = input_config['dims']
        
        if self.mode == 'along_axis':
            self.axis = input_config['gravity_along_axis']
            self._check_direction = [0] * len(self.shape)
            self._check_direction[self.axis] = input_config['gravity_direction_along_axis']
        

    def fall(self, board, position):
        if self.mode == 'along_axis':
            return self._down_along_the_axis(board, position)

    def _down_along_the_axis(self, board, positon):
        old_positon = tuple(util.copy_list(positon))
        for distance in range(1, self.shape[self.axis] - positon[self.axis]):
            new_position = util.add_coordinates(old_positon, self._check_direction)
            if board[new_position] != 0:
                break
            else:
                old_positon = new_position
        return old_positon