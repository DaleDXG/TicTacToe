import baselines.method_MCTS as method_MCTS

class Agent():
    
    def __init__(self, input_config=None):
        assert input_config != None, ('A InputConfig is needed before create an environment.')

        self.state_size = input_config['dims']
        self.action_size = input_config['action_size']

        self.cpuct = input_config['cpuct']

        self.MCTSsimulations = None
        self.model = None

        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []
    
    def simulate(self):
        pass

    def act(self, state, tau):
        pass

    def get_preds(self, state):
        pass

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):
        pass

    def getAV(self, tau):
        pass

    def chooseAction(self, pi, values, tau):
        pass

    def replay(self, ltmemory):
        pass

    def predict(self, state):
        pass

    def buildMCTS(self, state):
        pass

    def changeRootMCTS(self, state):
        pass
