
from model_baselines.exploration.base import ExplorationBase

class EpsilonGreedy(ExplorationBase):

    def __init__(self, input_config=None):
        self.epsilon_min = input_config.epsilon_greedy #
        self.epsilon_decrement = input_config.epsilon_greedy_decrement

    def choose_action(self):
        pass

    def step_update(self):
        pass