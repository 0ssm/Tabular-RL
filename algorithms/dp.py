import numpy as np

def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    num_state = env.observation_space.n
    V = np.zeros(num_state)
    while True:
        delta = 0
        for state in range(num_state):
            old_value = V[state]
            new_value = 0
            for action in range(env.action_space.n):
                action_probability = policy[state, action]
                for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                    if terminated:
                        future_value = 0
                    else:
                        future_value = V[next_state]
                    new_value += action_probability * probability * (reward + gamma * future_value)
            V[state] = new_value
            delta = max(delta, abs(old_value - new_value))
        if delta < theta:
            break
    return V


def policy_improvement(env, V, gamma=0.9):
    num_state = env.observation_space.n
    num_action = env.action_space.n
    policy = np.zeros((num_state, num_action))
    for state in range(num_state):
        action_value = np.zeros(num_action)
        for action in range(num_action):
            for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                if terminated:
                    future_value = 0
                else:
                    future_value = V[next_state]
                action_value[action] += probability * (reward + gamma * future_value)
        best_action = np.argmax(action_value)
        policy[state, best_action] = 1.0
    return policy


def policy_iteration(env, gamma=0.9, theta=1e-6):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.ones((num_states, num_actions)) / num_actions
    while True:
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy = policy_improvement(env, V, gamma=gamma)
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    return policy, V


def value_iteration(env, gamma=0.9, theta=1e-6):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    V = np.zeros(num_states)
    while True:
        delta = 0
        for state in range(num_states):
            old_value = V[state]
            action_value = np.zeros(num_actions)
            for action in range(num_actions):
                for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                    if terminated:
                        future_value = 0
                    else:
                        future_value = V[next_state]
                    action_value[action] += probability * (reward + gamma * future_value)
            V[state] = np.max(action_value)
            delta = max(delta, abs(old_value - V[state]))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma=gamma,)
    return policy, V