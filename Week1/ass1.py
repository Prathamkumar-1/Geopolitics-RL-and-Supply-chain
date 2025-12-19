# Omitted some parts from the code so it won't run till you fill them up

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment setup
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -----------------------------
# Hyperparameters
# -----------------------------
alpha = 0.1      # learning rate
gamma = 0.95     # discount factor
epsilon = 1.0    # initial exploration
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 20000
max_steps = 100

# -----------------------------
# Q-table initialization
# -----------------------------
Q = np.zeros((n_states, n_actions))

# -----------------------------
# Logging
# -----------------------------
episode_rewards = []
success_rate = []

success_count = 0
eval_window = 500

# -----------------------------
# Q-learning loop
# -----------------------------
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    for step in range(max_steps):

        # Implement ε-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        # Q-learning update

        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        if done:
            if reward == 1:
                success_count += 1
            break
    # Explore epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    # Track success rate 
    if (episode + 1) % eval_window == 0:
        success_rate.append(success_count / eval_window)
        success_count = 0

env.close()

