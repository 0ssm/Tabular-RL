# First-Visit MC Prediction implementation works on Blackjack

import numpy as np

def generate_episode(env, policy):
    episode = []
    state, info = env.reset()
    while True:
        action = np.random.choice(env.action_space.n, p=policy(state))
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        if terminated or truncated:
            break
        state = next_state
    return episode


def mc_prediction(env, policy, gamma = 1.0,  num_episode = 10_000):
    v = {}
    returns = {}
    for _ in range(num_episode):
        episode = generate_episode(env, policy)
        first_index = {}
        for index, (state, action, reward) in enumerate(episode):
            if state not in first_index:
                first_index[state] = index
        G = 0
        for index in range(len(episode) -1, -1, -1):
            state, action, reward = episode[index]
            G = reward + gamma * G
            if first_index[state] == index:
                if state not in returns:
                    returns[state] = []
                returns[state].append(G)
                v[state] = np.mean(returns[state])


    return v