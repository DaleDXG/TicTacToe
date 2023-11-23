
class RandomAgent():

    def __init__(self, env_type, env_select_action=None, env_random_step_function=None):
        self.env_type = env_type
        self.env_select_action = env_select_action
        self.env_random_step_function = env_random_step_function
        self.history = []
        self.score_list = []

    # use a lot of '*args, **kwargs' to avoid raising error

    def select_action(self, *args, **kwargs):
        if self.env_select_action is not None:
            return self.env_select_action()

    def save_model(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        if self.env_random_step_function is not None:
            return self.env_random_step_function()