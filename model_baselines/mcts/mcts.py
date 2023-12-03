
import util
import Config

import math
import numpy as np



class Node:

    index = 0

    def __init__(self, state, player_turn):
        self.state = state
        self.player_turn = player_turn
        self.id = Node.index
        Node.index += 1
        self.edges = [] # edge 添加的时候是action:edge的键值对字典

    def is_leaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True
        


class Edge:

    def __init__(self, inNode, outNode, prior, action):
        self.id = str(inNode.id) + '|' + str(outNode.id)
        self.inNode = inNode
        self.outNode = outNode
        self.player_turn = inNode.player_turn
        self.action = action

        self.stats =  {
                    'N': 0,
                    'W': 0,
                    'Q': 0,
                    'P': prior,
                }



class MCTS:

    def __init__(self, root, c_puct):
        self.root = root
        self.tree = {} # dictionary
        self.c_puct = c_puct
        self.add_node(root)
    
    def __len__(self):
        return len(self.tree)
    
    def add_node(self, node):
        self.tree[node.id] = node

    def move_to_leaf(self, env_simulation):

        util.logger_mcts.info('------MOVING TO LEAF------')

        # 此次寻找叶子节点所走过的edge
        breadcrumbs = []
        current_node = self.root

        done = False
        reward = 0

        while not current_node.is_leaf():

            util.logger_mcts.info('PLAYER TURN...%d', current_node.player_turn)

            maxQU = -99999

            if current_node == self.root:
                # dale 这是不是要用到epsilon-greedy呢？
                epsilon = Config.EPSILON
                # dale 此句暂不清楚
                nu = np.random.dirichlet([Config.ALPHA] * len(current_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            # Nb 是当前节点总的访问次数
            Nb = 0
            for action, edge in current_node.edges:
                Nb = Nb + edge.stats['N']

            for index, (action, edge) in enumerate(current_node.edges):
                
                # 此 U 未公式的后半部分
                U = self.c_puct * \
                    ((1-epsilon) * edge.stats['P'] + epsilon * nu[index]) \
                    * math.sqrt(Nb) / (1 + edge.stats['N'])
                
                Q = edge.stats['Q']
                
                # 输出信息，暂时不用懂
                util.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
					, action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[index],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[index] )
					, np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))

                if Q + U > maxQU:
                    maxQU = Q + U
                    # 当前MCTS选出的action，循环最后，找到最大的
                    simulation_action = action
                    simulation_edge = edge

            util.logger_mcts.info('action with highest Q + U...%d', simulation_action)
            
            # dale 这个地方有问题：
            # 它选完了以后，实际模拟了一下，但是过于确信得到的state、reward等信息和此前一样，
            # 没有更新Node和Edge的任何信息，那仿真的意义是什么？只为了拿reward和done？
            # 并且这也是只针对回合制的，先不管
            # 开始需要留意了，这是模拟simulation的过程，既要调用到env，又需要现存再env，不能改了env的值
            state, reward, done, info = env_simulation.step(simulation_action)
            current_node = simulation_edge.outNode
            breadcrumbs.append(simulation_edge)

        util.logger_mcts.info('DONE...%d', done)

        return current_node, reward, done, breadcrumbs
    
    
    def back_fill(self, leaf, reward, breadcrumbs):
        util.logger_mcts.info('------DOING BACKFILL------')

        current_player = leaf.player_turn

        for edge in breadcrumbs:
            player_turn = edge.player_turn
            if player_turn == current_player:
                direction = 1
            else:
                direction = -1

            # 这样说来，之前负值也被访问很多次，其实是合理的，因为对手的回合，会采用 吗？
            # 不是，这个direction的作用，是将负的reward变成正值，这样selection阶段只要一直选最大值就可以了
            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + reward * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            util.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
                , reward * direction
                , player_turn
                , edge.stats['N']
                , edge.stats['W']
                , edge.stats['Q']
                )
            
            util.logger_mcts.info('\n' + str(edge.outNode.state))






