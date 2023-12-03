
from model_baselines.exploration.base import ExplorationBase



class MCTS(ExplorationBase):
    # root
    # root_stats
    # tree
    # c_puct 平衡探索与收敛的超参数

    def __init__(self, root, cpuct):
        self.root = root
        self.root_stats = Edge(Node('', None), root, None, None)
        self.tree = {}
        self.countID = 0
        self.cpuct = cpuct
        self.addNode(root)
    
    def __len__(self):
        return len(self.tree)
    
    def addNode(self, node):
        self.tree[node.id] = node

    def moveToLeaf(self):
        print('start move to leaf')
        currentNode = self.root
        edge_parent = self.root_stats
        
        # 记录探索路径
        breadcrumbs = [self.root_stats]
        
        while not currentNode.isLeaf():
            child_nodes_unvisited = []
            edgeMaxQU = edge_parent
            MaxQU = -9999
            for idx, edge in enumerate(currentNode.edges):
                if edge.stats['N'] == 0:
                    child_nodes_unvisited.append((idx, edge))
                else:
                    QU = self.UCTSelection(edge_parent, edge)
                    if QU > MaxQU:
                        MaxQU = QU
                        edgeMaxQU = edge
            if len(child_nodes_unvisited) > 0:
                randIdx = randint(0, len(child_nodes_unvisited) - 1)
                edge_parent = currentNode.edges[randIdx]
            else:
                edge_parent = edgeMaxQU
                # currentNode = edgeMaxQU.outNode
                # breadcrumbs.append(edgeMaxQU)
            currentNode = edge_parent.outNode
            breadcrumbs.append(edge_parent)
            
        print('find leaf ' + str(currentNode.id) + ', path len = ' + str(len(breadcrumbs)))
        return currentNode, breadcrumbs
    
    def UCTSelection(self, edge_parent, edge):
        # formula
        if edge.stats['N'] == 0:
            QU = float_info.max
        else:
            QU = edge.stats['Q'] + self.cpuct * np.sqrt(np.log(edge_parent.stats['N']) / edge.stats['N'])
        return QU
    
    def Expansion(self, currentNode, actions):
        for action in actions:
            self.countID += 1
            nextBoard = copy.copy(currentNode.state.board)
            nextBoard[action] = currentNode.state.playerTurn
            nextNode = Node(self.countID, Game.GameState_TicTacToe(nextBoard, -currentNode.state.playerTurn))
            currentNode.edges.append(Edge(currentNode, nextNode, None, action))
            print('expand ' + str(nextNode.id))
    
    def Simulation_forBoardGame_RandomStrategy(self, game:Game.BoardGameBase, actions = []):
        while not (game.checkTermination() or len(actions) <= 0):
            selectedAction = actions.pop(randint(0, len(actions) - 1))
            game.addPiece(selectedAction // 3, selectedAction % 3)
        print('end simulation')
        return game.checkWinning()
    
    def Backpropagation(self, leaf, value, breadcrumbs):
        print('start backfill')
        currentPlayer = leaf.state.playerTurn # breadcrumbs[breadcrumbs.__len__].outNode.playerTurn
        for edge in breadcrumbs:
            if edge.outNode.state.playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] += 1
            edge.stats['W'] += value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
        print('end backfill')



class Node:
    # state
    # id
    # edges

    def __init__(self, id, state):
        # 因为对手的应对未知，感觉上状态应该存：
        # 父节点的状态，和要执行的动作
        self.state = state # playerTurn action observation
        self.id = id
        self.edges = []
        # 0 表示为被探索
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
    # stats = {N, W, Q , P}

    def __init__(self, inNode, outNode, prior, action):
        if inNode == None or inNode == '':
            self.id = '|' + str(outNode.id)
        else:
            self.id = str(inNode.id) + '|' + str(outNode.id)
        self.inNode = inNode
        self.outNode = outNode
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