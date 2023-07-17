# from BoardGameBase import BoardGameBase
# from TicTacToe_env_core import TicTacToe_env_core
# from TicTacToe_env import TicTacToe_env

from gym.envs.registration import register

register(
    id='TicTacToe-v0',
    entry_point='TicTacToe.envs:TicTacToe_env',
    max_episode_steps=300
)