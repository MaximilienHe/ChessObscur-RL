"""
logger.py — TensorBoard + console logging.

CHANGES from v1:
- Log all game/* metrics in console output
- Log rollout diagnostic metrics
- Better formatting
"""
import os
import time
from torch.utils.tensorboard import SummaryWriter
from typing import Dict


class Logger:
    def __init__(self, log_dir: str = "runs", run_name: str = None):
        if run_name is None:
            run_name = f"chess_obscur_{int(time.time())}"
        self.log_path = os.path.join(log_dir, run_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_path)
        self.start_time = time.time()
        self.last_log_time = time.time()
        print(f"[logger] TensorBoard: {self.log_path}")

    def log_scalars(self, metrics: Dict[str, float], step: int):
        for key, val in metrics.items():
            self.writer.add_scalar(key, val, step)

    def log_console(self, step: int, total_steps: int, metrics: Dict[str, float],
                    games_completed: int = 0, fps: float = 0.0):
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        progress = step / max(total_steps, 1) * 100

        parts = [
            f"step={step:,}",
            f"progress={progress:.1f}%",
            f"fps={fps:.0f}",
            f"games={games_completed:,}",
            f"time={hours:.1f}h",
        ]

        # PPO losses
        for key in ["loss/policy", "loss/value", "loss/entropy", "ppo/approx_kl"]:
            if key in metrics:
                parts.append(f"{key.split('/')[-1]}={metrics[key]:.4f}")

        # CHANGED: log ALL game metrics
        for key in ["game/win_rate", "game/draw_rate", "game/avg_length",
                     "game/white_win_rate", "game/black_win_rate"]:
            if key in metrics:
                parts.append(f"{key.split('/')[-1]}={metrics[key]:.3f}")

        # NEW: rollout diagnostics
        gc = metrics.get("rollout/games_completed", 0)
        if gc > 0:
            parts.append(f"games_in_rollout={int(gc)}")

        avg_reward = metrics.get("rollout/mean_reward", None)
        if avg_reward is not None:
            parts.append(f"mean_rew={avg_reward:.4f}")

        print(" | ".join(parts))

    def close(self):
        self.writer.close()