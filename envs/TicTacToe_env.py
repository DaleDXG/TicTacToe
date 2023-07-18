import gym
from gym import spaces
import pygame

from envs.TicTacToe_env_core import TicTacToe_env_core



class TicTacToe_env(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, input_config=None):
        assert input_config != None ('A InputConfig is needed before create an environment.')
        self._config = input_config
        self.flag_self_play_view = input_config['flag_self_play_view']

        self._env = TicTacToe_env_core(input_config)
        
        self.observation_space = {
            "board": self.map
        }
        self.action_space = spaces.Discrete(self.len_map)
    
        
    def init_screen(self):
        self.window_size = 512
    
    def reset(self):
        self._env.reset()

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        if self.flag_self_play_view and self._env.previous_player != 1:
            info['origin_board'] = observation['board']
            observation['board'] = self.self_play_map_view()
    
    def self_play_map_view(self):
        return ((self._env.map.copy() - (1 + self._env.previous_player)) % self._env.num_players + 1) * (self._env.map != 0)
