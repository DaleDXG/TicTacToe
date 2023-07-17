


class GravitySetting(object):

    modes = ('along_axis', 'toward_axis', 'toward_point')

    def __init__(self, input_config=None):
        assert input_config['gravity_mode'] in GravitySetting.modes (
            'unknow mode is provided'
        )
        self.mode = input_config['gravity_mode']
        self.shape = input_config['dims']
        
        if self.mode == 'along_axis':
            self.axis = input_config['gravity_along_axis']
        

    def fall(self, board, position):
        if self.mode == 'along_axis':
            return self._down_along_the_axis(board, position)

    def _down_along_the_axis(self, board, positon):
        pass