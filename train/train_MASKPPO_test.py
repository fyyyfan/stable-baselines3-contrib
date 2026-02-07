"""
MaskablePPO minimal working example (SB3-Contrib)
- A tiny custom env where some actions are invalid depending on the state.
- Demonstrates:
  1) env.action_masks() interface
  2) training with MaskablePPO
  3) evaluation using maskable evaluation utilities

Install:
  pip install -U stable-baselines3 sb3-contrib gymnasium numpy
Run:
  python train_maskableppo_toy.py
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


class MaskedLineEnv(gym.Env):
    """
    A super simple env:
    - State: position pos in [0, 4]
    - Actions (Discrete(3)):
        0 = left
        1 = stay
        2 = right
    - Invalid actions:
        - At pos == 0: action 0 (left) is invalid
        - At pos == 4: action 2 (right) is invalid

    Goal: reach pos==4 quickly.
    Reward:
      +1 when reaching pos==4, else -0.01 per step (small time penalty)
    Episode ends when pos==4 or max_steps reached.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 50, seed: int | None = 0):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=4.0, shape=(1,), dtype=np.float32)

        self.max_steps = max_steps
        self.np_random = None
        self._seed = seed

        self.pos = 0
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed if seed is not None else self._seed)
        self.pos = 0
        self.steps = 0
        obs = np.array([self.pos], dtype=np.float32)
        info = {}
        return obs, info

    def action_masks(self) -> np.ndarray:
        # True = valid, False = invalid
        mask = np.ones(self.action_space.n, dtype=bool)
        if self.pos == 0:
            mask[0] = False  # cannot go left
        if self.pos == 4:
            mask[2] = False  # cannot go right
        # Safety: ensure at least one valid action
        if not mask.any():
            mask[1] = True
        return mask

    def step(self, action: int):
        self.steps += 1

        # OPTIONAL: you can enforce validity here too
        # (MaskablePPO should already avoid invalid actions when masks are passed correctly)
        if not self.action_masks()[action]:
            # If an invalid action slips through (e.g., you forgot to pass masks at inference),
            # penalize and do not move.
            reward = -1.0
        else:
            if action == 0:   # left
                self.pos = max(0, self.pos - 1)
            elif action == 2: # right
                self.pos = min(4, self.pos + 1)
            reward = -0.01

        terminated = (self.pos == 4)
        if terminated:
            reward = 1.0

        truncated = (self.steps >= self.max_steps)

        obs = np.array([self.pos], dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info


def quick_manual_rollout(model: MaskablePPO, env: gym.Env, n_steps: int = 20):
    """Demonstrate inference with masks (important!)."""
    obs, _ = env.reset()
    for t in range(n_steps):
        masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    env = MaskedLineEnv(max_steps=30, seed=0)

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        verbose=1,
        tensorboard_log="./maskableppo_toy_logs/",
        # You can tweak these, but defaults should work for this toy
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(total_timesteps=30_000, log_interval=10)
    model.save("maskableppo_toy")

    # Mask-aware evaluation (do NOT use stable_baselines3.common.evaluation.evaluate_policy here)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=50,
        deterministic=True,
    )
    print(f"[Eval] mean_reward={mean_reward:.3f} +/- {std_reward:.3f}")

    # Quick rollout to verify inference path uses masks
    quick_manual_rollout(model, env, n_steps=50)

    print("Done. Open TensorBoard with:")
    print("  tensorboard --logdir=./train/maskableppo_toy_logs/")
