from env import CircularTrackEnv
from ppo import PPOAgent

def main():
    env = CircularTrackEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)

    episodes = 500

    for ep in range(episodes):
        state, _ = env.reset()
        done = False

        states, actions = [], []
        rewards, log_probs, values = [], [], []

        ep_reward = 0
        max_steps = 300
        steps = 0
        while not done and steps < max_steps:
            steps += 1
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state
            ep_reward += reward

        agent.update((states, actions, rewards, log_probs, values))

        if ep % 50 == 0:
            print(f"Episode {ep}, Total Reward: {ep_reward:.2f}")


if __name__ == "__main__":
    main()
