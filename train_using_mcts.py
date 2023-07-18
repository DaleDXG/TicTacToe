import envs
from envs.Config import InputConfig
from envs.TicTacToe_env import TicTacToe_env

input_config = InputConfig(flag_self_play_view=True, flag_compute_used_left=True)
env = TicTacToe_env(input_config)

obs, reward, done, info = env.step(0,0)
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step(0,1)
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step(1,1)
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step(0,2)
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step(2,2)
env._env.display_console()
print(obs['board'])
obs, reward, done, info = env.step(1,0)
env._env.display_console()
print(obs['board'])