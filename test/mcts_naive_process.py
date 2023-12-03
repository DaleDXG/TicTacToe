
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import Config
import util
from envs.TicTacToe_env import TicTacToe_env
from model_baselines.mcts.mcts import MCTS

import copy

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


def process(env, mcts):
    countLoop = 0
    while countLoop < 5000:
        print('round ' + str(countLoop))
        currentNode, breadcrumbs = mcts.moveToLeaf()
        actionsLeft = []
        for idx, cell in enumerate(currentNode.state.board):
            if cell == 0:
                actionsLeft.append(idx)
        if not env.checkTermination(currentNode.state.board):
            mcts.Expansion(currentNode, actionsLeft)
        # for edge in breadcrumbs:
        #     game_sim.addPiece(edge.action // 3, edge.action % 3)
        game_sim = copy.deepcopy(env)
        game_sim.map = copy.copy(currentNode.state.board)
        value = mcts.Simulation_forBoardGame_RandomStrategy(game_sim, actionsLeft)
        mcts.Backpropagation(currentNode, value, breadcrumbs)
        countLoop += 1
    root_stats = mcts.root_stats
    wfs(root_stats)


if __name__ == "__main__":
    
    input_config_env = Config.InputConfig_Env(size=3, num_dim=2,
                                          flag_self_play_view=True,
                                          flag_compute_used_left=True)
    
    env = TicTacToe_env(input_config_env)
    init_state = env.reset
    mcts = MCTS(init_state, util.C)

    # T3 = Game.TicTacToe()
    # root = mc.Node(0, Game.GameState_TicTacToe([0] * 9, 1))

    process(env, mcts)