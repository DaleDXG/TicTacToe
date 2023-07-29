import envs
from Config import InputConfig_Env
from envs.TicTacToe_env import TicTacToe_env

input_config = InputConfig_Env(size=3, num_dim=2, flag_self_play_view=True)
# input_config = InputConfig_Env(size=7, num_dim=2, num_in_a_row=4, gravity_mode='along_axis', flag_self_play_view=True)
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

if __name__ == "__main__":
    print(type({'a':1}))