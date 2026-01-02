import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.fc(state)
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.clip_eps = 0.2

        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean, value = self.model(state)

        std = torch.ones_like(mean) * 0.5
        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.detach().numpy(), log_prob, value

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self, memory):
        states, actions, rewards, log_probs, values = memory

        returns = torch.FloatTensor(self.compute_returns(rewards))
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        old_log_probs = torch.stack(log_probs).detach()
        values = torch.stack(values).squeeze().detach()


        advantages = returns - values

        for _ in range(4):
            mean, new_values = self.model(states)
            dist = torch.distributions.Normal(mean, 0.5)
            new_log_probs = dist.log_prob(actions).sum(axis=1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
