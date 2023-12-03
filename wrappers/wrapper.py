
import gym
from envs.TicTacToe_env import TicTacToe_env



class DirectObservation(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._wrppers_with_support = {
            'CheckpointRewardWrapper', 'FrameStack', 'GetStateWrapper',
            'SingleAgentRewardWrapper', 'SingleAgentObservationWrapper',
            'SMMWrapper', 'PeriodicDumpWriter', 'Simple115StateWrapper',
            'PixelsStateWrapper'
        }

    def reset(self):
        # return TicTacToe_env.reset()