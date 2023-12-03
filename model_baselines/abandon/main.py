
# abandent

import baselines.method_MCTS as mc
import copy

# actions_left = [
#     (0, 0),
#     (0, 1),
#     (0, 2),
#     (1, 0),
#     (1, 1),
#     (1, 2),
#     (2, 0),
#     (2, 1),
#     (2, 2),
# ]
actions_left = [0,1,2,3,4,5,6,7,8]



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
    # mcts = mc.MCTS(root, 1)
    # mcts.Simulation_forBoardGame_RandomStrategy(T3, actions_left)
    # print(actions_left)
    # print('winning ' + str(T3.checkWinning()))
    # T3.display_console()
    process(T3, root)

    # T3 = Game.TicTacToe()
    # for i in range(3):
    #     for j in range(3):
    #         print('winning ' + str(T3.checkWinning()))
    #         print('full ' + str(T3.checkFull()))
    #         T3.display_console()
    #         T3.addPiece(i, j)
    #         print('')
    # print('winning ' + str(T3.checkWinning()))
    # print('full ' + str(T3.checkFull()))
    # T3.display_console()
    
    # mcts = mc.MCTS(mc.Node(0, Game.GameState_TicTacToe([0] * 9, 1)), 1)
    # a = Game.TicTacToe()
    # b = copy.deepcopy(a)
    # mcts.Simulation_forBoardGame_RandomStrategy(b, [0,1,2,3,4,5,6,7,8])
    # print(a.map)
    # print(b.map)