"""
league.py — League training: maintain a pool of past checkpoints as opponents.

v8: Breaks the Nash draw equilibrium by forcing the agent to exploit weaknesses
in older versions of itself, rather than converging to a symmetric draw strategy.

Architecture:
- LeaguePool stores N past network snapshots (CPU, no grad).
- During rollout, ~30% of envs use a random opponent from the pool.
- When it's the opponent's turn in league envs, the opponent network picks the action.
- The learning network still evaluates all states for PPO (log_probs, values).
"""
import random
import torch
import torch.nn as nn
from typing import Optional, List

from config import Config
from model.network import ChessObscurNetwork

SnapshotEntry = tuple[int, dict[str, torch.Tensor]]


class LeaguePool:
    """Manages a pool of past network snapshots for league training."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_checkpoints = cfg.league_max_checkpoints
        self.snapshot_interval = cfg.league_checkpoint_interval
        self._snapshots: List[SnapshotEntry] = []
        self._opponent_net: Optional[ChessObscurNetwork] = None
        self._last_snapshot_step = 0

    def _target_snapshot_steps(self, latest_step: int) -> List[int]:
        ages = [0]
        if self.max_checkpoints > 1:
            ages.append(self.snapshot_interval)
        if self.max_checkpoints > 2:
            ages.append(2 * self.snapshot_interval)

        age = 5 * self.snapshot_interval
        while len(ages) < self.max_checkpoints:
            ages.append(age)
            age *= 2

        return sorted(max(0, latest_step - target_age) for target_age in ages[:self.max_checkpoints])

    def _score_snapshot_set(self, snapshots: List[SnapshotEntry]) -> int:
        steps = sorted(step for step, _ in snapshots)
        targets = self._target_snapshot_steps(steps[-1])
        return sum(abs(step - target) for step, target in zip(steps, targets))

    def _trim_snapshots(self):
        if len(self._snapshots) <= self.max_checkpoints:
            return

        latest_step = self._snapshots[-1][0]
        best_snapshots = None
        best_score = None

        for remove_idx in range(len(self._snapshots)):
            candidate = self._snapshots[:remove_idx] + self._snapshots[remove_idx + 1:]
            if candidate[-1][0] != latest_step:
                continue

            score = self._score_snapshot_set(candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_snapshots = candidate

        if best_snapshots is not None:
            self._snapshots = best_snapshots

    def _create_opponent_net(self) -> ChessObscurNetwork:
        """Create an opponent network on the same device as training."""
        net = ChessObscurNetwork(
            obs_planes=self.cfg.obs_planes,
            num_filters=self.cfg.num_filters,
            num_res_blocks=self.cfg.num_res_blocks,
            policy_head_filters=self.cfg.policy_head_filters,
            value_head_hidden=self.cfg.value_head_hidden,
            total_actions=self.cfg.total_actions,
            value_head_channels=self.cfg.value_head_channels,
            use_attention=self.cfg.use_attention,
            attention_heads=self.cfg.attention_heads,
        ).to(self.cfg.device)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    def maybe_snapshot(self, network: nn.Module, global_step: int):
        """Take a snapshot of the current network if enough steps have passed."""
        if self.max_checkpoints <= 0:
            return
        if global_step - self._last_snapshot_step < self.snapshot_interval:
            return
        if global_step < self.snapshot_interval:
            return

        # Save state_dict on same device (avoids CPU→GPU transfer on each load)
        state_dict = network.state_dict()
        # Strip _orig_mod. prefix from torch.compile()
        snapshot_state = {}
        for k, v in state_dict.items():
            clean_key = k.replace("_orig_mod.", "")
            snapshot_state[clean_key] = v.detach().clone()

        self._snapshots.append((global_step, snapshot_state))
        self._trim_snapshots()

        self._last_snapshot_step = global_step
        print(f"[league] Snapshot taken at step {global_step:,} "
              f"(pool size: {len(self._snapshots)})")

    def has_opponents(self) -> bool:
        return len(self._snapshots) > 0

    def get_random_opponent(self) -> ChessObscurNetwork:
        """Load a random past snapshot into the opponent network and return it."""
        if not self._snapshots:
            raise RuntimeError("No snapshots in league pool")

        if self._opponent_net is None:
            self._opponent_net = self._create_opponent_net()

        _, snapshot_state = random.choice(self._snapshots)
        self._opponent_net.load_state_dict(snapshot_state)
        return self._opponent_net

    @property
    def pool_size(self) -> int:
        return len(self._snapshots)
