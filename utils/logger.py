"""
logger.py — TensorBoard + console logging.

CHANGES v5:
- Removed parry/enemy_capture logs (illegal move removed)
- Removed parry diagnostic logs (could_move, skip_when_*)
- Parry stats: self_capture_rate, good_move_rate, skip_rate only
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

        # Game metrics
        for key in ["game/win_rate", "game/draw_rate", "game/avg_length",
                     "game/white_win_rate", "game/black_win_rate"]:
            if key in metrics:
                parts.append(f"{key.split('/')[-1]}={metrics[key]:.3f}")

        gc = metrics.get("rollout/games_completed", 0)
        if gc > 0:
            parts.append(f"games_in_rollout={int(gc)}")

        avg_reward = metrics.get("rollout/mean_reward", None)
        if avg_reward is not None:
            parts.append(f"mean_rew={avg_reward:.4f}")

        # Parry stats (3 outcomes: skip, good_move, self_capture)
        for key in ["parry/self_capture_rate", "parry/good_move_rate", "parry/skip_rate"]:
            if key in metrics:
                parts.append(f"{key.split('/')[-1]}={metrics[key]:.3f}")

        # Capture quality
        if "capture/high_attacker_rate" in metrics:
            parts.append(f"high_atk_rate={metrics['capture/high_attacker_rate']:.3f}")

        # Check escape
        if "check/3rd_attempt_move_rate" in metrics:
            parts.append(f"3rd_chk_move={metrics['check/3rd_attempt_move_rate']:.3f}")

        print(" | ".join(parts))

    def close(self):
        self.writer.close()