
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import util

def dqn_mlp_process(env_reset, env_step, env_random_step, agent, input_config):

    episodes = input_config.max_episodes # 1000
    for i in range(episodes):
        # print('run env reset')
        init_state = env_reset()
        # print('run agent episode starting reset')
        agent.reset(init_state)
        while True:
            # print('run agent.select_action')
            action = agent.select_action(agent.current_state)
            next_s, reward, done, info = env_step(action)
            # print('run agent.step')
            s_r, reward_r, done_r, info_r = env_random_step()
            agent.step(s_r, reward, done, info)
            if done:
                break
        # if np.mean(score_list[-10:]) > -160:
        #     agent.save_model()
        #     break
        
    agent.save_model()
    plt.plot(agent.history, color='green')
    plt.show()
    # env.close()


colour_set = ['green', 'blue', 'yellow', 'red', 'purple']

def dqn_inturn_multiplayer_process(env_reset, env_step, agents, input_config):

    episodes = input_config.max_episodes # 1000
    model_save_iter = 200
    num_agents = len(agents)
    assert num_agents > 0, ('There is no agent participating the game.')

    flag_static_memory = input_config.flag_static_memory
    # idx_previous_agent = -1
    for i in range(episodes):
        util.logger_env.info('\n-----New episode-----')
        step = 0
        done = False
        idx_current_agent = 0
        if num_agents == 1:
            idx_next_agent = 0
        else:
            idx_next_agent = 1
        count_down = num_agents
        cache_replay = deque(maxlen=num_agents)
        init_state = env_reset()
        newest_state = init_state
        for agent in agents:
            util.logger_env.info('agent reset')
            agent.reset(init_state)
        while True:
            if not done:
                action = agents[idx_current_agent].select_action(newest_state) # agents[idx_current_agent].current_state
            else:
                action = 0
            newest_state, reward, done, info = env_step(action)
            cache_replay.append([newest_state, reward, done, info])
            if step > num_agents - 2: # >= num_agents - 1:
                util.logger_env.info('agent ' + str(idx_next_agent) + ' step')
                agents[idx_next_agent].step(newest_state, cache_replay[0][1], cache_replay[0][2], cache_replay[0][3])

            # idx_previous_agent = idx_current_agent
            idx_current_agent = idx_next_agent
            idx_next_agent = (idx_next_agent + 1) % len(agents)
            step += 1
            if done:
                if count_down == 0:
                    break
                count_down -= 1
        
        # if np.mean(score_list[-10:]) > -160:
        #     agent.save_model()
        #     break
        if episodes % model_save_iter == 0:
            agents[0].save_model('model_dqn_' + str(episodes) + '.h5')
    for i, agent in enumerate(agents):
        plt.plot(agent.history, color=colour_set[i])
    plt.show()