def self_play(env, agent1, agent2, num_episode):
    for _ in range(num_episode):
        env.reset()
        done = False
        current_agent = agent1
        while not done:
            current_state = env.self_play_map_view() # get_state()
            action = current_agent.choose_action(current_state)
            observation, reward, done, info =  env.step(action // 3, action % 3)
            next_state = env.get_state()
            current_agent.update_policy(current_state, action)
            current_agent = agent2 if current_agent == agent1 else agent1