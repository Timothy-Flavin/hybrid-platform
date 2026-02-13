# Hybrid Platform

A Gymnasium environment for the Platform domain, featuring a parameterized (hybrid) action space. This environment is widely used for benchmarking Reinforcement Learning algorithms that must handle both discrete action choices and continuous parameters simultaneously.

This is a modern fork of the original [gym-platform](https://github.com/cycraig/gym-platform), fully updated to use the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API and modern `numpy`.

![gym-platform](ressources/platform_domain.png)

## Installation

You can install the package directly from PyPI:

```bash
pip install hybrid-platform
```

## Usage

```python
import gymnasium as gym
import gym_platform

# Create the environment
env = gym.make('Platform-v0', render_mode='human')

obs, info = env.reset()

for _ in range(100):
    # Action structure: (discrete_action_index, (param_0, param_1, param_2))
    # where param_X is an array of shape (1,)
    
    # Example: RUN (0) with speed 15.0
    action = (0, ([15.0], [0.0], [0.0]))
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Action Space

The action space is `spaces.Tuple((spaces.Discrete(3), spaces.Tuple((spaces.Box(1), spaces.Box(1), spaces.Box(1)))))`. It consists of:
1.  **Discrete(3)**: The high-level action choice.
    *   `0`: **Run** (Move horizontally)
    *   `1`: **Hop** (Small jump to clear gaps)
    *   `2`: **Leap** (Large jump to clear larger gaps)
2.  **Tuple(Box(1), Box(1), Box(1))**: The continuous parameters for each action.
    *   Parameter 0 (Run): Speed, range `[0, 30]`
    *   Parameter 1 (Hop): Power, range `[0, 720]`
    *   Parameter 2 (Leap): Power, range `[0, 430]`

When taking a step, the environment selects the parameter corresponding to the chosen discrete action index and ignores the others.

## Observation Space

The observation space is a `spaces.Tuple` containing:
1.  **Box(9,)**: The state vector (scaled).
    *   Basic features: `[player_x, player_vx, enemy_x, enemy_dx]`
    *   Platform features: `[platform_width_1, platform_width_2, gap, platform_position, height_diff]`
2.  **Discrete(200)**: The current time step (used for time awareness).

## Original Credits
*   Original Platform domain by Warwick Masson et al.
*   Original gym-platform implementation by [Craig Bester](https://github.com/cycraig).
