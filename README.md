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
