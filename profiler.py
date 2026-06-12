"""
Profiler for Platform-v0: runs 100 high-level steps (actions) headless,
reports wall-clock steps/sec, and dumps a cProfile breakdown.
"""

import cProfile
import pstats
import io
import time
import numpy as np
import sys
import os

# Suppress pygame display output
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import gym_platform  # noqa: F401 — registers Platform-v0

NUM_FRAMES = 100

MIN_PARAMS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
MAX_PARAMS = np.array([30.0, 720.0, 430.0], dtype=np.float32)


def make_action(rng: np.random.Generator):
    act_idx = rng.integers(0, 3)
    params = tuple(
        np.array([rng.uniform(MIN_PARAMS[i], MAX_PARAMS[i])], dtype=np.float32)
        for i in range(3)
    )
    return (int(act_idx), params)


def run_frames(env, rng, n=NUM_FRAMES):
    """Execute n high-level steps, resetting on termination."""
    obs, _ = env.reset()
    total_substeps = 0
    episodes = 0

    for _ in range(n):
        action = make_action(rng)
        obs, reward, terminated, truncated, info = env.step(action)
        # obs[1] is the number of internal sub-steps taken this action
        total_substeps += obs[1]
        if terminated or truncated:
            obs, _ = env.reset()
            episodes += 1

    return total_substeps, episodes


def main():
    env = gym.make("Platform-v0")  # no render_mode → headless
    rng = np.random.default_rng(42)

    print(f"Profiling Platform-v0 for {NUM_FRAMES} high-level steps (headless)…\n")

    # ── 1. Wall-clock timing ──────────────────────────────────────────────────
    # Warm-up: one reset to JIT-compile numba if present
    env.reset()
    env.step(make_action(rng))

    rng = np.random.default_rng(42)  # reset rng for reproducible run
    t0 = time.perf_counter()
    total_substeps, episodes = run_frames(env, rng, NUM_FRAMES)
    elapsed = time.perf_counter() - t0

    steps_per_sec = NUM_FRAMES / elapsed
    substeps_per_sec = total_substeps / elapsed

    print("── Timing ──────────────────────────────────────────────")
    print(f"  High-level steps : {NUM_FRAMES}")
    print(f"  Internal substeps: {total_substeps}  (avg {total_substeps/NUM_FRAMES:.1f} per action)")
    print(f"  Episodes done    : {episodes}")
    print(f"  Elapsed          : {elapsed:.3f} s")
    print(f"  Steps/sec        : {steps_per_sec:.1f}  (high-level actions)")
    print(f"  Substeps/sec     : {substeps_per_sec:.1f}  (internal physics ticks)")
    print()

    # ── 2. cProfile breakdown ─────────────────────────────────────────────────
    env.reset()
    rng = np.random.default_rng(42)

    pr = cProfile.Profile()
    pr.enable()
    run_frames(env, rng, NUM_FRAMES)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)  # top 20 functions by cumulative time
    print("── cProfile (top 20 by cumulative time) ─────────────────")
    print(buf.getvalue())

    env.close()


if __name__ == "__main__":
    main()
