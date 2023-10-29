import model_baselines.mcts.mcts as mcts



def wfs(root_stats):
    print('start writting')
    _queue = [root_stats]
    with open("output.txt", "a") as file:
        # file.write('数据录入\n')
        while len(_queue) > 0:
            currentEdge = _queue.pop(0)
            file.write(str(currentEdge.inNode.id) +' '+ str(currentEdge.outNode.id) +' '+ str(currentEdge.outNode.state.playerTurn) +' '+ str(currentEdge.outNode.state.board) +' '+ str(currentEdge.action) +' '+ str(currentEdge.stats) + '\n')
            print('writen: ' + str(currentEdge.outNode.id) +' '+ str(currentEdge.outNode.state.playerTurn) +' '+ str(currentEdge.outNode.state.board) +' '+ str(currentEdge.action) +' '+ str(currentEdge.stats))
            for edge in currentEdge.outNode.edges:
                _queue.append(edge)
        
# 脑子有点乱，组织一下带selfplay的过程
# 玩家1，selection、expansion、simulation、backfill
# 玩家2正分版本
# 玩家2的回合，selection（找分高的），expansion，simulation（-1玩家2赢了加分，输了负分）、backfill（赢了加分、输了减分）
# 需要至少两个树甚至更多，然后再整合起来
# 玩家2负分版本
# 玩家2的回合，selection（找负分的，绝对值高的，数值小的），expansion，simulation（-1玩家2赢了减分，输了加分）、backfill（赢了减分、输了加分）

# 五月18 prob:
# 1. board containing pieces more than expect
# 2. nagetive score get many visit

class Agent:

    def __init__(self, input_config):
        assert input_config != None, ('A InputConfig is needed before create an environment.')

        self.state_size = input_config['dims']
        self.action_size = input_config['action_size']

        self.c_puct = input_config['c_puct']

        self.MCTSsimulations = None
        self.model = None

        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def step(self, state):
        pass
    
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
        self.root = mcts.Node(state)
        self.tree = mcts.MCTS(self.root, self.c_puct)

    def changeRootMCTS(self, state):
        pass
    
def process(game, root):
    mcts = mc.MCTS(root, 1)
    countLoop = 0
    while countLoop < 5000:
        print('round ' + str(countLoop))
        currentNode, breadcrumbs = mcts.moveToLeaf()
        actionsLeft = []
        for idx, cell in enumerate(currentNode.state.board):
            if cell == 0:
                actionsLeft.append(idx)
        if not game.checkTermination(currentNode.state.board):
            mcts.Expansion(currentNode, actionsLeft)
        # for edge in breadcrumbs:
        #     game_sim.addPiece(edge.action // 3, edge.action % 3)
        game_sim = copy.deepcopy(game)
        game_sim.map = copy.copy(currentNode.state.board)
        value = mcts.Simulation_forBoardGame_RandomStrategy(game_sim, actionsLeft)
        mcts.Backpropagation(currentNode, value, breadcrumbs)
        countLoop += 1
    root_stats = mcts.root_stats
    wfs(root_stats)

if __name__ == "__main__":
    # pass
    
    T3 = Game.TicTacToe()
    root = mc.Node(0, Game.GameState_TicTacToe([0] * 9, 1))

    process(T3, root)