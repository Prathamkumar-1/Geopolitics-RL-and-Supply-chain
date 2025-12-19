import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make(
    "FrozenLake-v1",
    map_name="6x6",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

num_episodes = 15000
max_steps = 100
eval_window = 500

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])


def train_q_learning():
    Q = np.zeros((n_states, n_actions))
    rewards, success_rate = [], []

    eps = epsilon
    success_count = 0

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # OFF-POLICY update
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

            if done:
                if reward == 1:
                    success_count += 1
                break

        eps = max(epsilon_min, eps * epsilon_decay)
        rewards.append(total_reward)

        if (ep + 1) % eval_window == 0:
            success_rate.append(success_count / eval_window)
            success_count = 0

    return Q, rewards, success_rate


def train_sarsa():
    Q = np.zeros((n_states, n_actions))
    rewards, success_rate = [], []

    eps = epsilon
    success_count = 0

    for ep in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, eps)
        total_reward = 0

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = epsilon_greedy(Q, next_state, eps)

            # ON-POLICY update
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                if reward == 1:
                    success_count += 1
                break

        eps = max(epsilon_min, eps * epsilon_decay)
        rewards.append(total_reward)

        if (ep + 1) % eval_window == 0:
            success_rate.append(success_count / eval_window)
            success_count = 0

    return Q, rewards, success_rate

Q_q, rewards_q, success_q = train_q_learning()
Q_s, rewards_s, success_s = train_sarsa()

plt.plot(success_q, label="Q-learning")
plt.plot(success_s, label="SARSA")
plt.xlabel("Evaluation Window")
plt.ylabel("Success Rate")
plt.title("Q-learning vs SARSA (FrozenLake 6x6)")
plt.legend()
plt.show()

def extract_policy(Q):
    return np.argmax(Q, axis=1)

policy_q = extract_policy(Q_q)
policy_s = extract_policy(Q_s)


def print_policy(policy, size=6):
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    grid = np.array([arrows[a] for a in policy]).reshape(size, size)
    print(grid)

print("Q-learning policy:")
print_policy(policy_q)

print("\nSARSA policy:")
print_policy(policy_s)
