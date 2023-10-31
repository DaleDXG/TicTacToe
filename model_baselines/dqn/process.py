
import numpy as np
import matplotlib.pyplot as plt

def dqn_process(env, agent, input_config):

    episodes = input_config.max_episodes # 1000
    score_list = []
    # agent = DQN()
    for i in range(episodes):
        s = env.reset()
        score = 0
        while True:
            a = agent.act(s)
            next_s, reward, done, _ = env.step(a)
            agent.remember(s, a, next_s, reward)
            agent.train()
            score += reward
            s = next_s
            if done:
                score_list.append(score)
                print('episode:', i, 'score:', score, 'max:', max(score_list))
                break
        if np.mean(score_list[-10:]) > -160:
            agent.save_model()
            break
        
    plt.plot(score_list, color='green')
    plt.show()
    env.close()

