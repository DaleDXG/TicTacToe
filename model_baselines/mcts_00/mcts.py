from random import randint
import numpy as np
from sys import float_info
import copy

import util
import Config

class Node:
    # state
    # id
    # edges
    index = 0

    def __init__(self, state, player_turn):
        # 因为对手的应对未知，感觉上状态应该存：
        # 父节点的状态，和要执行的动作
        self.state = state # playerTurn action observation
        # 自增id
        self.id = Node.index
        Node.index += 1

        self.player_turn = player_turn

        self.edges = []
        # 0 表示未被探索
        # self.flag = 0
    
    def isLeaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge():
    # id
    # inNode # where the edge start from
    # outNode # where the edge link to
    # action
    # stats = {N, W, Q, P}

    def __init__(self, inNode, outNode, prior, action):
        if inNode == None or inNode == '':
            self.id = '|' + str(outNode.id)
        else:
            self.id = str(inNode.id) + '|' + str(outNode.id)
        self.inNode = inNode
        self.outNode = outNode
        self.player_turn = inNode.player_turn
        self.action = action

        # edge.stats['N'] = edge.stats['N'] + 1
        # edge.stats['W'] = edge.stats['W'] + value * direction
        # edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior
        }


class MCTS():
    # root
    # root_stats
    # tree
    # c_puct 平衡探索与收敛的超参数

    def __init__(self, init_state, cpuct):
        self.root = Node(init_state, 1)
        # Node('', None)
        # self.root_stats = Edge(None, self.root, None, None)
        self.tree = {}
        # self.countID = 0
        self.cpuct = cpuct
        self.addNode(self.root)
    
    def __len__(self):
        return len(self.tree)
    
    def addNode(self, node):
        self.tree[node.id] = node

    
    def moveToLeaf(self):

        util.logger_mcts.info('------MOVING TO LEAF------')

        breadcrumbs = []
        currentNode = self.root

        done = 0
        value = 0

        while not currentNode.isLeaf():

            util.logger_mcts.info('PLAYER TURN...%d', currentNode.player_turn)

            maxQU = -99999

            if currentNode == self.root:
                epsilon = Config.EPSILON
                nu = np.random.dirichlet([Config.ALPHA] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):

                U = self.cpuct * \
                    ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
                    np.sqrt(Nb) / (1 + edge.stats['N'])
                    
                Q = edge.stats['Q']

                util.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
                    , action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
                    , np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge

            util.logger_mcts.info('action with highest Q + U...%d', simulationAction)

            newState, value, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)

        util.logger_mcts.info('DONE...%d', done)

        return currentNode, value, done, breadcrumbs
    

    def backFill(self, leaf, value, breadcrumbs):
        util.logger_mcts.info('------DOING BACKFILL------')

        currentPlayer = leaf.player_turn

        for edge in breadcrumbs:
            playerTurn = edge.player_turn
            if playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            util.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
                , value * direction
                , playerTurn
                , edge.stats['N']
                , edge.stats['W']
                , edge.stats['Q']
                )

            # edge.outNode.state.render(util.logger_mcts)



    # def moveToLeaf(self):
    #     print('start move to leaf')
    #     currentNode = self.root
    #     edge_parent = self.root_stats
        
    #     # 记录探索路径
    #     breadcrumbs = [self.root_stats]
        
    #     while not currentNode.isLeaf():
    #         child_nodes_unvisited = []
    #         edgeMaxQU = edge_parent
    #         MaxQU = -9999
    #         for idx, edge in enumerate(currentNode.edges):
    #             if edge.stats['N'] == 0:
    #                 child_nodes_unvisited.append((idx, edge))
    #             else:
    #                 QU = self.UCTSelection(edge_parent, edge)
    #                 if QU > MaxQU:
    #                     MaxQU = QU
    #                     edgeMaxQU = edge
    #         if len(child_nodes_unvisited) > 0:
    #             randIdx = randint(0, len(child_nodes_unvisited) - 1)
    #             edge_parent = currentNode.edges[randIdx]
    #         else:
    #             edge_parent = edgeMaxQU
    #             # currentNode = edgeMaxQU.outNode
    #             # breadcrumbs.append(edgeMaxQU)
    #         currentNode = edge_parent.outNode
    #         breadcrumbs.append(edge_parent)
            
    #     print('find leaf ' + str(currentNode.id) + ', path len = ' + str(len(breadcrumbs)))
    #     return currentNode, breadcrumbs
    
    # def UCTSelection(self, edge_parent, edge):
    #     # formula
    #     if edge.stats['N'] == 0:
    #         QU = float_info.max
    #     else:
    #         QU = edge.stats['Q'] + self.cpuct * np.sqrt(np.log(edge_parent.stats['N']) / edge.stats['N'])
    #     return QU
    
    # def Expansion(self, currentNode, actions):
    #     for action in actions:
    #         self.countID += 1
    #         nextBoard = copy.copy(currentNode.state.board)
    #         nextBoard[action] = currentNode.state.playerTurn
    #         nextNode = Node(self.countID, Game.GameState_TicTacToe(nextBoard, -currentNode.state.playerTurn))
    #         currentNode.edges.append(Edge(currentNode, nextNode, None, action))
    #         print('expand ' + str(nextNode.id))
    
    # def Simulation_forBoardGame_RandomStrategy(self, game:Game.BoardGameBase, actions = []):
    #     while not (game.checkTermination() or len(actions) <= 0):
    #         selectedAction = actions.pop(randint(0, len(actions) - 1))
    #         game.addPiece(selectedAction // 3, selectedAction % 3)
    #     print('end simulation')
    #     return game.checkWinning()
    
    # def Backpropagation(self, leaf, value, breadcrumbs):
    #     print('start backfill')
    #     currentPlayer = leaf.state.playerTurn # breadcrumbs[breadcrumbs.__len__].outNode.playerTurn
    #     for edge in breadcrumbs:
    #         if edge.outNode.state.playerTurn == currentPlayer:
    #             direction = 1
    #         else:
    #             direction = -1

    #         edge.stats['N'] += 1
    #         edge.stats['W'] += value * direction
    #         edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
    #     print('end backfill')