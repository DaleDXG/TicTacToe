
from gym.envs.registration import register

register(
    id='TicTacToe',
    entry_point='TicTacToe.envs:TicTacToe_env',
    max_episode_steps=300
)