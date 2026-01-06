# Geopolitics-RL-and-Supply-chain

This repository contains my work for two weekly assignments(for now) from the WIDS project **Geopolitics-RL-and-Supply-chain**  
The focus across both weeks was on understanding algorithm behavior of RL techniques.

---

## Week 1 — Q-learning and SARSA on FrozenLake

### Objective
Implement tabular **Q-learning** and **SARSA** on the `FrozenLake-v1` (8×8, slippery) environment and compare their behavior experimentally and theoretically.

### What was done
- Implemented tabular Q-learning with ε-greedy exploration
- Studied:
  - fixed vs decaying ε
  - sensitivity to learning rate (α)
  - sensitivity to discount factor (γ)
- Implemented SARSA using the same environment and hyperparameters
- Compared Q-learning and SARSA in terms of:
  - success rate
  - learned policy
  - behavior near holes (risky transitions)

### Key observations
- **Q-learning (off-policy)** learns faster but can be overly optimistic in stochastic environments.
- **SARSA (on-policy)** learns more conservative policies, avoiding risky states near holes.
- In FrozenLake, SARSA often prefers safer paths due to on-policy updates under exploration.

---

## Week 2 — Circular Driving Environment with PPO

### Objective
Design a simple annular (circular) driving environment and solve it using **Proximal Policy Optimization (PPO)**.

### What was done
- Built a custom Gym-style environment with:
  - state = (x, y, velocity, heading)
  - continuous actions (steering, acceleration)
  - simple kinematic dynamics
  - reward based on track adherence and progress
- Implemented a minimal PPO agent using an Actor–Critic architecture
- Trained the agent and debugged common RL issues such as:
  - infinite episodes (added max step limits)
  - PPO backpropagation errors (detached old policy tensors)
- Used simple matplotlib-based visualization to inspect trajectories locally

---

## Running the Code (High-level)

```bash
# Week 1
python ass1.py  # Q-learning policy
python assignment2.py  # SARSA policy

# Week 2
python train.py        # PPO training

```
### Supply Chain Background

Introductory videos on supply chain management and global logistics helped build intuition about how real-world supply chains operate across multiple stages (manufacturers, distributors, retailers). Case studies and documentaries, such as those by WSJ and analyses of Amazon’s logistics network, were useful in understanding the scale, delays, and coordination challenges present in global supply systems.

More focused resources on **multi-echelon inventory systems** and the **bullwhip effect** highlighted how small changes in demand can amplify across different levels of a supply chain. The Beer Game and its simulations provided a concrete example of how delayed information and decentralized decision-making can lead to inefficiencies and instability. These ideas are particularly relevant for modeling supply chains as interacting agents rather.

### Multi-Agent Reinforcement Learning

The MARL book was used as a high-level reference to understand the types of problems addressed in multi-agent settings, such as coordination, competition, and non-stationarity.Basic **Game theory**  was also studied to understand concepts like equilibria and incentive alignment.

Lecture videos and research talks on MARL, including seminar-style presentations, helped differentiate multi-agent learning from standard RL. These resources emphasized why assumptions that hold in single-agent environments (such as stationarity of the environment) often break down when multiple learning agents interact.

Still the progress on Multi-Agent RL is going on .
