
import numpy as np
import random

import util
import model_baselines.mcts.mcts as mcts


        
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

        self.c_puct = input_config.c_puct

        self.mcts = None

        # self.state_size = input_config.dims
        # self.action_size = input_config.action_size
        # self.MCTSsimulations = None
        # self.model = None

        # self.train_overall_loss = []
        # self.train_value_loss = []
        # self.train_policy_loss = []
        # self.val_overall_loss = []
        # self.val_value_loss = []
        # self.val_policy_loss = []

    def reset(self):
        pass

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
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1/tau)
            values[action] = edge.stats['Q']
        
        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_index = np.random.multinomial(1, pi)
            action = np.where(action_index==1)[0][0]
        
        value = values[action]

        return action, value


    def replay(self, ltmemory):
        pass

    def predict(self, X):
        preds = self.model.predict(X)
        return preds

    def buildMCTS(self, init_state):
        util.logger_env.info('-----build_MCTS-----')
        self.mcts = mcts.MCTS(init_state, self.c_puct)

    def changeRootMCTS(self, id):
        util.logger_env.info('-----change_MCTS_root-----')
        self.mcts.root = self.mcts.tree[id]



# def wfs(root_stats):
#     print('start writting')
#     _queue = [root_stats]
#     with open("output.txt", "a") as file:
#         # file.write('数据录入\n')
#         while len(_queue) > 0:
#             currentEdge = _queue.pop(0)
#             file.write(str(currentEdge.inNode.id) +' '+ str(currentEdge.outNode.id) +' '+ str(currentEdge.outNode.state.playerTurn) +' '+ str(currentEdge.outNode.state.board) +' '+ str(currentEdge.action) +' '+ str(currentEdge.stats) + '\n')
#             print('writen: ' + str(currentEdge.outNode.id) +' '+ str(currentEdge.outNode.state.playerTurn) +' '+ str(currentEdge.outNode.state.board) +' '+ str(currentEdge.action) +' '+ str(currentEdge.stats))
#             for edge in currentEdge.outNode.edges:
#                 _queue.append(edge)

# def process(game, root):
#     mcts = mc.MCTS(root, 1)
#     countLoop = 0
#     while countLoop < 5000:
#         print('round ' + str(countLoop))
#         currentNode, breadcrumbs = mcts.moveToLeaf()
#         actionsLeft = []
#         for idx, cell in enumerate(currentNode.state.board):
#             if cell == 0:
#                 actionsLeft.append(idx)
#         if not game.checkTermination(currentNode.state.board):
#             mcts.Expansion(currentNode, actionsLeft)
#         # for edge in breadcrumbs:
#         #     game_sim.addPiece(edge.action // 3, edge.action % 3)
#         game_sim = copy.deepcopy(game)
#         game_sim.map = copy.copy(currentNode.state.board)
#         value = mcts.Simulation_forBoardGame_RandomStrategy(game_sim, actionsLeft)
#         mcts.Backpropagation(currentNode, value, breadcrumbs)
#         countLoop += 1
#     root_stats = mcts.root_stats
#     wfs(root_stats)

# if __name__ == "__main__":
#     # pass
    
#     T3 = Game.TicTacToe()
#     root = mc.Node(0, Game.GameState_TicTacToe([0] * 9, 1))

#     process(T3, root)