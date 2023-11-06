
import numpy as np
import matplotlib.pyplot as plt

def dqn_mlp_process(env_reset, env_step, env_random_step, agent, input_config):

    episodes = input_config.max_episodes # 1000
    for i in range(episodes):
        print('run env reset')
        init_state = env_reset()
        print('run agent episode starting reset')
        agent.step_reset(init_state)
        while True:
            print('run agent.select_action')
            action = agent.select_action(agent.current_state)
            next_s, reward, done, info = env_step(action)
            print('run agent.step')
            agent.step(next_s, reward, done, info)
            s_r, reward_r, done_r, info_r = env_random_step()
            agent.opponent_turn_update(s_r, reward_r)
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

def dqn_mlp_selfplay_process(env_reset, env_step, agents, input_config):

    episodes = input_config.max_episodes # 1000
    index = 0
    for i in range(episodes):
        for agent in agents:
            agent.step_reset(env_reset)
        while True:
            current_agent = agents[index]
            print(current_agent)
            done = current_agent.step(env_step)
            if done:
                break
            index = (index + 1) % len(agents)
        # if np.mean(score_list[-10:]) > -160:
        #     agent.save_model()
        #     break
        
    agents[0].save_model()
    plt.plot(agents[0].history, color='green')
    plt.show()