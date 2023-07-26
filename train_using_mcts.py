import envs
from Config import InputConfig
from envs.TicTacToe_env import TicTacToe_env

input_config = InputConfig(size=3, num_dim=2, flag_self_play_view=True)
# input_config = InputConfig(size=7, num_dim=2, gravity_mode='along_axis', flag_self_play_view=True)
env = TicTacToe_env(input_config)

obs, reward, done, info = env.step((0,0))
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step((0,1))
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step((1,1))
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step((0,2))
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step((2,2))
env._env.display_console()
print(obs['board'])
# obs, reward, done, info = env.step((1,0))
# env._env.display_console()
# print(obs['board'])