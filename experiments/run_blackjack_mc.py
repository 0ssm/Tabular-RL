import gymnasium as gym
import numpy as np

from algorithms.monte_carlo import mc_prediction


def policy(state):

    player_sum, dealer_card, usable_ace = state
    if player_sum < 20:
        return np.array([0.0, 1.0])
    return np.array([1.0, 0.0])


def main():

    env = gym.make("Blackjack-v1")
    num_episodes = 10_000
    V = mc_prediction(
        env=env,
        policy=policy,
        gamma=1.0,
        num_episode=num_episodes,
    )
    print("Monte Carlo Blackjack Prediction")
    print("--------------------------------")
    print("Number of episodes:", num_episodes)
    print("Number of states visited:", len(V))
    print("\nSample state values:")

    for state, value in list(V.items())[:20]:
        print(f"{state} -> {value:.4f}")


    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_episodes):
        episode = []
        state, info = env.reset()
        while True:
            action = np.random.choice(
                env.action_space.n,
                p=policy(state)
            )
            next_state, reward, terminated, truncated, info = env.step(action)

            episode.append((state, action, reward))
            if terminated or truncated:
                break
            state = next_state
        final_reward = episode[-1][2]
        if final_reward == 1:
            wins += 1
        elif final_reward == -1:
            losses += 1
        else:
            draws += 1

    print("\nPolicy Performance")
    print("------------------")
    print("Total episodes:", num_episodes)
    print("Wins:", wins)
    print("Losses:", losses)
    print("Draws:", draws)

    print(f"Win rate:   {wins / num_episodes:.2%}")
    print(f"Loss rate:  {losses / num_episodes:.2%}")
    print(f"Draw rate:  {draws / num_episodes:.2%}")

    env.close()


if __name__ == "__main__":
    main()