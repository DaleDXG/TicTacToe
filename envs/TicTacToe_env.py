import gym

from TicTacToe_env_core import TicTacToe_env_core



class TicTacToe_env(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, size=3, num_dim=2, dim=None, num_in_a_row=3, gravity_setting=None, players=(1, 1)):
        self._env = TicTacToe_env_core(size, num_dim, dim, num_in_a_row, gravity_setting, players)
        
        
    def init_screen(self):
        self.window_size = 512
        