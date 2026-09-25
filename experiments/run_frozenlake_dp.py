# this was for running episode and test the sucessful and failure

import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from algorithms.dp import policy_iteration

def main():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="rgb_array",)
    policy, V = policy_iteration(env, gamma=0.9,)
    print("Optimal Value Function:")
    print(np.round(V.reshape(8, 8), 7))
    state, info = env.reset()
    frames = []
    frames.append(env.render())
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.argmax(policy[state])
        state, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
    env.close()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save GIF into result
    gif_path = results_dir / "frozenlake_policy.gif"
    imageio.mimsave(gif_path, frames, duration=0.8,loop=0,)

    print()
    print("Episode finished!")
    print("Reward:", reward)
    print("GIF saved to:", gif_path.resolve())

main()