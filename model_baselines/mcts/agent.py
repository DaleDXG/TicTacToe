
import numpy as np
import random

import model_baselines.mcts.mcts as mcts

import Config
import util
import time
import copy

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl




class User():
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, tau):
        action = input('Enter your chosen action: ')
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        NN_value = None
        return (action, pi, value, NN_value)



class Agent():
    def __init__(self, name, state_size, action_size, mcts_simulations, c_puct, model):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.c_puct = c_puct

        self.MCTSsimulations = mcts_simulations # default is 50
        self.model = model

        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []


    #
    def set_env(self, env):
        self.env = env

    def get_allowed_actions(self):
        return self.env._env.leftover_positions


    # 这个函数就不该叫simulate，因为它和MCTS的步骤simulation不对应
    def simulate(self, env_simulation):

        util.logger_mcts.info('ROOT NODE...%s', self.mcts.root.id)
        util.logger_mcts.info('\n' + str(self.mcts.root.state))
        util.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.player_turn)

        ##### MOVE THE LEAF NODE
        leaf, reward, done, breadcrumbs = self.mcts.move_to_leaf(env_simulation)
        util.logger_mcts.info('\n' + str(leaf.state))

        ##### EVALUATE THE LEAF NODE
        reward, breadcrumbs = self.evaluate_leaf(leaf, reward, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.back_fill(leaf, reward, breadcrumbs)


    def act(self, state, player_turn, tau):

        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state, player_turn)
        else:
            self.changeRootMCTS(state)

        #### run the simulation
        # dale added
        # original_core = self.env.get_state()
        for sim in range(self.MCTSsimulations):
            util.logger_mcts.info('***************************')
            util.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            util.logger_mcts.info('***************************')
            # 这里需要传入模拟
            self.simulate(copy.deepcopy(self.env))
            # dale added
            # self.env.set_state(original_core)

        # 这个函数叫getAV是真tm的恶趣味
        #### get action values
        # 两个结果都是数组
        pi, values = self.get_action_values(1)

        ####pick the action
        action, value = self.choose_action(pi, values, tau)

        # dale 这又是一个很烦的GameState调用，想办法解决一下
        # next_state, _, _ = state.take_action(action)
        next_state, _, _, _ = copy.deepcopy(self.env).step(action)
        next_state = next_state['board']

        NN_value = -self.get_preds(next_state)[0]

        util.logger_mcts.info('ACTION VALUES...%s', pi)
        util.logger_mcts.info('CHOSEN ACTION...%d', action)
        util.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        util.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

        return (action, pi, value, NN_value)


    def get_preds(self, state):
        #predict the leaf
        inputToModel = self.model.convertToModelInput(state) # np.array([self.model.convertToModelInput(state)])

        preds = self.model.predict(inputToModel)
        # 下面这一连串是在倒腾什么呢？
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]

        logits = logits_array[0]

        # dale 又是GameState，这里传一个函数即可
        allowed_actions = self.get_allowed_actions() # state.allowedActions

        mask = np.ones(logits.shape,dtype=bool)
        mask[allowed_actions] = False
        logits[mask] = -100

        #SOFTMAX
        # 报错再说
        odds = np.exp(logits)
        probs = odds / np.sum(odds) ###put this just before the for?

        return ((value, probs, allowed_actions))


    def evaluate_leaf(self, leaf, value, done, breadcrumbs):

        util.logger_mcts.info('------EVALUATING LEAF------')

        if done == 0:
            
            # 这个state是那个恶心的...
            value, probs, allowed_actions = self.get_preds(leaf.state)
            util.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.player_turn, value)

            probs = probs[allowed_actions]

            for idx, action in enumerate(allowed_actions):
                # dale 又是GameState
                newState, _, _ = leaf.state.take_action(action)
                if newState.id not in self.mcts.tree:
                    node = mcts.Node(newState)
                    self.mcts.add_node(node)
                    util.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
                else:
                    node = self.mcts.tree[newState.id]
                    util.logger_mcts.info('existing node...%s...', node.id)

                newEdge = mcts.Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))
                
        else:
            util.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.player_turn, value)

        return ((value, breadcrumbs))


        
    def get_action_values(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        
        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1/tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def choose_action(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0]

        value = values[action]

        return action, value

    def replay(self, ltmemory):
        util.logger_mcts.info('******RETRAINING MODEL******')


        for i in range(Config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(Config.BATCH_SIZE, len(ltmemory)))

            training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])
                                , 'policy_head': np.array([row['AV'] for row in minibatch])} 

            fit = self.model.fit(training_states, training_targets, epochs=Config.EPOCHS, verbose=1, validation_split=0, batch_size = 32)
            util.logger_mcts.info('NEW LOSS %s', fit.history)

            self.train_overall_loss.append(round(fit.history['loss'][Config.EPOCHS - 1],4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][Config.EPOCHS - 1],4)) 
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][Config.EPOCHS - 1],4)) 

        plt.plot(self.train_overall_loss, 'k')
        plt.plot(self.train_value_loss, 'k:')
        plt.plot(self.train_policy_loss, 'k--')

        plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()
        time.sleep(1.0)

        print('\n')
        self.model.printWeightAverages()

    def predict(self, inputToModel):
        preds = self.model.predict(inputToModel)
        return preds

    def buildMCTS(self, state, player_turn):
        util.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mcts.Node(state, player_turn)
        self.mcts = mcts.MCTS(self.root, self.c_puct)

    def changeRootMCTS(self, id):
        util.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', id, self.name)
        self.mcts.root = self.mcts.tree[id]
		