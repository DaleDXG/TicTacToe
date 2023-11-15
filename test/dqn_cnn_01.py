import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


import Config
import util
from envs.TicTacToe_env import TicTacToe_env
from build_network_functions import build_network_00 as build
from model_baselines.dqn.agent02 import DQN
from model_baselines.dqn.process import dqn_inturn_multiplayer_process


env_config = Config.InputConfig_Env(size=3, num_dim=2,
                                    flag_self_play_view=True,
                                    flag_compute_used_left=True)
env = TicTacToe_env(env_config)

model_config = Config.InputConfig_Method(shape_layers=(9,9,9,9),
                                         max_episodes=1000000,
                                         epsilon_greedy=0.05,
                                         epsilon_greedy_decrement=0.001,
                                         function_build_cnn = build.build_cnn_00,
                                         flag_static_memory=True,
                                         weighted_replay_queue=True)
# model_config = Config.InputConfig_Method(shape_layers=(9,9,9,9),
#                                          flag_static_memory=True)
agent_01 = DQN(model_config)
agent_02 = DQN(model_config)

def wrap_env_reset_mlp_3t():
    observation = env.reset()
    return util.flatten_list(observation['board'])

def wrap_env_step_mlp_3t(action):
    observation, reward, done, info = env.step(action)
    state = util.flatten_list(observation['board'])
    return state, reward, done, info

def wrap_env_random_3t():
    observation, reward, done, info = env.random_policy()
    state = util.flatten_list(observation['board'])
    return state, reward, done, info

dqn_inturn_multiplayer_process(wrap_env_reset_mlp_3t, wrap_env_step_mlp_3t, [agent_01, agent_02], model_config)

env.close()



