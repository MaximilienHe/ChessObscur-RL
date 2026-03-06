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
import os
import copy
import random
import torch
import torch.nn as nn
from typing import Optional, List

from config import Config
from model.network import ChessObscurNetwork


class LeaguePool:
    """Manages a pool of past network snapshots for league training."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_checkpoints = cfg.league_max_checkpoints
        self.snapshot_interval = cfg.league_checkpoint_interval
        self._snapshots: List[dict] = []  # list of state_dicts (on CPU)
        self._opponent_net: Optional[ChessObscurNetwork] = None
        self._last_snapshot_step = 0

    def _create_opponent_net(self) -> ChessObscurNetwork:
        """Create an opponent network on the same device as training."""
        net = ChessObscurNetwork(
            obs_planes=self.cfg.obs_planes,
            num_filters=self.cfg.num_filters,
            num_res_blocks=self.cfg.num_res_blocks,
            policy_head_filters=self.cfg.policy_head_filters,
            value_head_hidden=self.cfg.value_head_hidden,
            total_actions=self.cfg.total_actions,
        ).to(self.cfg.device)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    def maybe_snapshot(self, network: nn.Module, global_step: int):
        """Take a snapshot of the current network if enough steps have passed."""
        if global_step - self._last_snapshot_step < self.snapshot_interval:
            return
        if global_step < self.snapshot_interval:
            return

        # Save state_dict to CPU
        state_dict = network.state_dict()
        # Strip _orig_mod. prefix from torch.compile()
        cpu_state = {}
        for k, v in state_dict.items():
            clean_key = k.replace("_orig_mod.", "")
            cpu_state[clean_key] = v.cpu().clone()

        self._snapshots.append(cpu_state)
        if len(self._snapshots) > self.max_checkpoints:
            self._snapshots.pop(0)

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

        snapshot = random.choice(self._snapshots)
        self._opponent_net.load_state_dict(snapshot)
        return self._opponent_net

    @property
    def pool_size(self) -> int:
        return len(self._snapshots)
