import envs
import util
from Config import InputConfig_Env
from Config import InputConfig_Method
from envs.TicTacToe_env import TicTacToe_env
from model_baselines.mlp import MLP

input_config_env = InputConfig_Env(size=3, num_dim=2, flag_self_play_view=True)
# input_config_env = InputConfig_Env(size=7, num_dim=2, num_in_a_row=4, gravity_mode='along_axis', flag_self_play_view=True)
env = TicTacToe_env(input_config_env)

input_config_method = InputConfig_Method(layers_size=(256,256,10), isFlatten=True)
mlp = MLP(input_config_method)
train_iter, test_iter = util.load_data_fashion_mnist(input_config_method.batch_size)
mlp.train(train_iter, test_iter)

