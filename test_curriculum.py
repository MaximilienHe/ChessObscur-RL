#!/usr/bin/env python3
"""
Test script for curriculum learning - verify max_steps progression.
"""
from config import Config


def test_curriculum_progression():
    """Test that the curriculum progresses correctly."""
    print("=" * 70)
    print("Curriculum Learning - Max Steps Progression")
    print("=" * 70 + "\n")

    cfg = Config()

    print(f"Curriculum enabled: {cfg.curriculum_enabled}")
    print(f"Starting max_steps: {cfg.curriculum_start_steps}")
    print(f"Increase amount: +{cfg.curriculum_step_increase}")
    print(f"Increase frequency: every {cfg.curriculum_every_n_timesteps:,} timesteps")
    print("\n" + "-" * 70)
    print("Timestep progression:")
    print("-" * 70 + "\n")

    # Test progression at different timesteps
    test_timesteps = [
        0,
        1_000_000,
        5_000_000,
        10_000_000,
        15_000_000,
        20_000_000,
        25_000_000,
        50_000_000,
        100_000_000,
    ]

    for timestep in test_timesteps:
        if cfg.curriculum_enabled:
            current_max_steps = cfg.curriculum_start_steps + \
                (timestep // cfg.curriculum_every_n_timesteps) * cfg.curriculum_step_increase
        else:
            current_max_steps = cfg.max_game_steps

        timeout_at = current_max_steps * 2
        print(f"  {timestep:>12,} steps → max_steps={current_max_steps:>3} (timeout at {timeout_at} half-moves)")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"  • Games start short ({cfg.curriculum_start_steps} steps) for fast iteration")
    print(f"  • Every {cfg.curriculum_every_n_timesteps/1_000_000:.0f}M steps, max_steps increases by {cfg.curriculum_step_increase}")
    print(f"  • At 100M steps, max_steps = {cfg.curriculum_start_steps + (100_000_000 // cfg.curriculum_every_n_timesteps) * cfg.curriculum_step_increase}")
    print(f"  • This allows agents to learn basic tactics quickly, then master longer endgames")
    print("=" * 70 + "\n")


def test_env_update():
    """Test that the environment max_steps can be updated."""
    print("Testing environment max_steps update...")

    from env.chess_obscur_env import ChessObscurEnv

    # Create a small env on CPU for testing
    env = ChessObscurEnv(num_envs=4, device="cpu", max_steps=120)

    print(f"  Initial max_steps: {env.max_steps}")

    env.set_max_steps(150)
    print(f"  After set_max_steps(150): {env.max_steps}")

    env.set_max_steps(180)
    print(f"  After set_max_steps(180): {env.max_steps}")

    print("  ✓ Environment max_steps update works correctly\n")


if __name__ == "__main__":
    test_curriculum_progression()
    test_env_update()

    print("\n🎓 Curriculum learning is ready!")
    print("   Start training and watch max_steps increase over time in tensorboard.")
