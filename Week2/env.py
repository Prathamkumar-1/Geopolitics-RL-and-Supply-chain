import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math

class CircularTrackEnv(gym.Env):
    """
    Simple circular driving environment.
    The physics is intentionally kept simple for learning purposes.
    """

    def __init__(self):
        super().__init__()

        # Track parameters
        self.r_inner = 4.0
        self.r_outer = 6.0
        self.r_mid = 5.0

        # Time step
        self.dt = 0.1
        self.wheelbase = 0.5

        # Action: [steering angle, acceleration]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # State: x, y, velocity, heading
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = self.r_mid
        self.y = 0.0
        self.v = 0.5
        self.theta = math.pi / 2  # facing along the track

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.x, self.y, self.v, self.theta], dtype=np.float32)

    def step(self, action):
        steering = action[0] * (math.pi / 4)     # max 45 degrees
        accel = action[1] * 0.2                  # small acceleration

        # Update velocity
        self.v = max(0.0, self.v + accel)

        # Update heading
        self.theta += (self.v / self.wheelbase) * math.tan(steering) * self.dt

        # Update position
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt

        r = math.sqrt(self.x**2 + self.y**2)

        # Reward design
        dist_penalty = -abs(r - self.r_mid)
        progress_reward = self.v
        reward = dist_penalty + 0.5 * progress_reward

        terminated = False

        # Off track
        if r < self.r_inner or r > self.r_outer:
            reward -= 20
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        r = math.sqrt(self.x**2 + self.y**2)
        print(f"x={self.x:.2f}, y={self.y:.2f}, r={r:.2f}, v={self.v:.2f}")
