
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import copy
import random
import numpy as np

import pickle

import Config
import util
from envs.TicTacToe_env import TicTacToe_env
from model_baselines.resnet.resnet import Residual_CNN
from model_baselines.mcts.mcts import MCTS
from model_baselines.mcts.memory import Memory
from model_baselines.mcts.agent01 import Agent

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


def playMatches(env, players, EPISODES, logger, turns_until_tau0, memory = None): # , goes_first = 0

    # env = Game()
    scores = {players[1].name:0, "drawn": 0, players[2].name:0}
    sp_scores = {'sp':0, "drawn": 0, 'nsp':0}
    points = {players[1].name:[], players[2].name:[]}
    env_core = env._env

    for e in range(EPISODES):

        logger.info('====================')
        logger.info('EPISODE %d OF %d', e+1, EPISODES)
        logger.info('====================')

        print (str(e+1) + ' ', end='')
        state = env.reset()
        
        done = 0
        turn = 0
        for player in players:
            player.mcts = None

        logger.info('\n' + str(env_core.map))
        input_dims = env_core.dims

        while done == 0:
            turn = turn + 1
    
            #### Run the MCTS algo and return an action
            if turn < turns_until_tau0:
                # dale 注意取得当前回合的方式 我的环境没有GameState.playerTurn，记录当前回合更不是通过toggle正负的方式实现的
                action, pi, MCTS_value, NN_value = players[env_core.current_player]['agent'].act(state, 1)
            else:
                action, pi, MCTS_value, NN_value = players[env_core.current_player]['agent'].act(state, 0)

            logger.info('action: %d', action)
            # for r in range(input_dims): # env.grid_shape[0]
            #     logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x,2)) for x in pi[env.grid_shape[1]*r : (env.grid_shape[1]*r + env.grid_shape[1])]])
            # logger.info('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(MCTS_value,2))
            # logger.info('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(NN_value,2))
            # logger.info('====================')

            ### Do the action
            state, reward, done, _ = env.step(action) #the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move
            
            # env.gameState.render(logger)
            logger.info('\n' + str(env_core.map))

            if memory != None:
                ####Commit the move to memory
                # memory.commit_stmemory(env.identities, state, pi)
                memory.stmemory.append({
                    'inputs': state,
                    'action_values': action_values,
                    'player_turn': player_turn
                    })

            if done == 1: 
                if memory != None:
                    #### If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if move['player_turn'] == env_core.current_player: # state.playerTurn:
                            move['value'] = reward
                        else:
                            move['value'] = -reward
                         
                    memory.commit_ltmemory()

                # 胜利
                if reward == 10:
                    logger.info('%s WINS!', players[state.playerTurn]['name'])
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1: 
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                # 失败
                elif reward == -5:
                    logger.info('%s WINS!', players[-state.playerTurn]['name'])
                    scores[players[-state.playerTurn]['name']] = scores[players[-state.playerTurn]['name']] + 1
               
                    if state.playerTurn == 1: 
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                else:
                    logger.info('DRAW...')
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])

    return (scores, memory, points, sp_scores)

# dale 临时救急
def identities(state, action_value):
    return state, action_value

if __name__ == "__main__":
    
    input_config_env = Config.InputConfig_Env(size=3, num_dim=2,
                                          flag_self_play_view=True,
                                          flag_compute_used_left=True)
    
    env = TicTacToe_env(input_config_env)

    init_state = env.reset
    mcts = MCTS(init_state, util.C)

    memory = Memory(Config.MEMORY_SIZE)

    # 输入是 grid_shape
    # 这里还不清楚为什么输入的shape需要在前面加一个 2
    current_NN = Residual_CNN(Config.REG_CONST, Config.LEARNING_RATE, util.flatten_list((2, input_config_env['dims'])), util.shape_to_num(input_config_env['dims']), Config.HIDDEN_CNN_LAYERS)
    best_NN = Residual_CNN(Config.REG_CONST, Config.LEARNING_RATE, util.flatten_list((2, input_config_env['dims'])), util.shape_to_num(input_config_env['dims']), Config.HIDDEN_CNN_LAYERS)

    best_player_version = 0
    best_NN.model.set_weights(current_NN.model.get_weights())

    # 输入是 state_size
    # Config的CPUCT=1，util的C=1/math.sqrt(2)
    current_player = Agent('current_player', input_config_env['dims'], util.shape_to_num(input_config_env['dims']), Config.MCTS_SIMS, util.C, current_NN)
    best_player = Agent('best_player', input_config_env['dims'], util.shape_to_num(input_config_env['dims']), Config.MCTS_SIMS, util.C, best_NN)

    iteration = 0

    while True:
        iteration += 1

        print('ITERATION NUMBER ' + str(iteration))

        util.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
        print('BEST PLAYER VERSION ' + str(best_player_version))

        ######## SELF PLAY ########
        players = {
            1: {"agent": best_player, "name":best_player.name},
            2: {"agent": best_player, "name":best_player.name}
            }
        print('SELF PLAYING ' + str(Config.EPISODES) + ' EPISODES...')
        _, memory, _, _ = playMatches(env, players, Config.EPISODES, util.logger_main, turns_until_tau0 = Config.TURNS_UNTIL_TAU0, memory = memory)
        print('\n')

        memory.clear_stmemory()
    
        if len(memory.ltmemory) >= Config.MEMORY_SIZE:

            ######## RETRAINING ########
            print('RETRAINING...')
            current_player.replay(memory.ltmemory)
            print('')

            if iteration % 5 == 0:
                pickle.dump( memory, open( "memory" + str(iteration).zfill(4) + ".p", "wb" ) )

            util.logger_memory.info('====================')
            util.logger_memory.info('NEW MEMORIES')
            util.logger_memory.info('====================')
            
            memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))
            
            for s in memory_samp:
                current_value, current_probs, _ = current_player.get_preds(s['state'])
                best_value, best_probs, _ = best_player.get_preds(s['state'])

                util.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
                util.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
                util.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
                util.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']]  )
                util.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in  current_probs])
                util.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in  best_probs])
                util.logger_memory.info('ID: %s', s['state'].id)
                util.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convertToModelInput(s['state']))

                # s['state'].render(util.logger_memory)
                util.logger_main.info('\n' + str(env._env.map))
                
            ######## TOURNAMENT ########
            players = {
                1: {"agent": best_player, "name":best_player.name},
                2: {"agent": current_player, "name":current_player.name}
                }
            print('TOURNAMENT...')
            scores, _, points, sp_scores = playMatches(best_player, current_player, Config.EVAL_EPISODES, util.logger_tourney, turns_until_tau0 = 0, memory = None)
            print('\nSCORES')
            print(scores)
            print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
            print(sp_scores)
            #print(points)

            print('\n\n')

            if scores['current_player'] > scores['best_player'] * Config.SCORING_THRESHOLD:
                best_player_version = best_player_version + 1
                best_NN.model.set_weights(current_NN.model.get_weights())
                best_NN.write(env.name, best_player_version)

        else:
            print('MEMORY SIZE: ' + str(len(memory.ltmemory)))
        
        # dale 加点结束条件，别让它一直跑，到后面修改到能随时结束再说
        if iteration > 500:
            break



    # T3 = Game.TicTacToe()
    # root = mc.Node(0, Game.GameState_TicTacToe([0] * 9, 1))

    # process(env, mcts)