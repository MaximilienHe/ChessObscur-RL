"""
ppo.py — Proximal Policy Optimization for Chess Obscur.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

from config import Config


class PPOTrainer:
    def __init__(self, network: nn.Module, config: Config):
        self.net = network
        self.cfg = config
        self.optimizer = optim.Adam(network.parameters(), lr=config.lr, eps=1e-5)
        self.lr_scheduler = None

        # Mixed precision support
        self.use_amp = config.use_amp and config.device == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.use_amp:
            print(f"[ppo] AMP enabled (mixed precision)")

        # ── NEW: track current entropy coef for logging ──
        self._current_entropy_coef = config.entropy_coef

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, N = rewards.shape
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(N, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_val * non_terminal - values[t]
            last_gae = delta + gamma * lam * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: Dict[str, torch.Tensor], global_step: int = 0) -> Dict[str, float]:
        """
        PPO update.
        
        CHANGED: accepts global_step to compute dynamic entropy_coef.
        """
        cfg = self.cfg

        # ── NEW: dynamic entropy coefficient decay ──
        entropy_coef = cfg.get_entropy_coef(global_step)
        self._current_entropy_coef = entropy_coef

        oom_errors = (torch.OutOfMemoryError,)
        if hasattr(torch, "AcceleratorError"):
            oom_errors = oom_errors + (torch.AcceleratorError,)

        with torch.no_grad():
            next_value = self.net.get_value(rollout["next_obs"])

        advantages, returns = self.compute_gae(
            rollout["rewards"], rollout["values"],
            rollout["dones"], next_value
        )

        T, N = rollout["rewards"].shape
        B = T * N

        b_obs = rollout["obs"].reshape(B, *rollout["obs"].shape[2:])
        b_actions = rollout["actions"].reshape(B)
        b_log_probs = rollout["log_probs"].reshape(B)
        b_advantages = advantages.reshape(B)
        b_returns = returns.reshape(B)
        b_values = rollout["values"].reshape(B)
        b_legal_masks = rollout["legal_masks"].reshape(B, -1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for epoch in range(cfg.ppo_epochs):
            perm = torch.randperm(B, device=b_obs.device)

            for start in range(0, B, cfg.minibatch_size):
                end = min(start + cfg.minibatch_size, B)
                mb_idx = perm[start:end]
                mb_size = end - start
                microbatch_size = max(1, min(cfg.microbatch_size, mb_size))

                while True:
                    mb_pg_loss = 0.0
                    mb_v_loss = 0.0
                    mb_entropy = 0.0
                    mb_clipfrac = 0.0
                    mb_approx_kl = 0.0

                    self.optimizer.zero_grad(set_to_none=True)

                    try:
                        for micro_start in range(0, mb_size, microbatch_size):
                            micro_end = min(micro_start + microbatch_size, mb_size)
                            micro_idx = mb_idx[micro_start:micro_end]
                            micro_count = micro_end - micro_start
                            micro_weight = micro_count / mb_size

                            mb_obs = b_obs[micro_idx]
                            mb_actions = b_actions[micro_idx]
                            mb_old_log_probs = b_log_probs[micro_idx]
                            mb_advantages = b_advantages[micro_idx]
                            mb_returns = b_returns[micro_idx]
                            mb_old_values = b_values[micro_idx]
                            mb_legal = b_legal_masks[micro_idx]

                            # Mixed precision forward pass
                            with torch.amp.autocast('cuda', enabled=self.use_amp):
                                _, new_log_probs, entropy, new_values = self.net.get_action_and_value(
                                    mb_obs, mb_legal, mb_actions
                                )

                                log_ratio = new_log_probs - mb_old_log_probs
                                ratio = torch.exp(log_ratio)

                                pg_loss1 = -mb_advantages * ratio
                                pg_loss2 = -mb_advantages * torch.clamp(
                                    ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps
                                )
                                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                                if cfg.clip_value > 0:
                                    v_clipped = mb_old_values + torch.clamp(
                                        new_values - mb_old_values, -cfg.clip_value, cfg.clip_value
                                    )
                                    v_loss1 = (new_values - mb_returns) ** 2
                                    v_loss2 = (v_clipped - mb_returns) ** 2
                                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                                else:
                                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                                entropy_loss = entropy.mean()

                                # ── CHANGED: use dynamic entropy_coef ──
                                loss = pg_loss + cfg.value_coef * v_loss - entropy_coef * entropy_loss

                            # Scaled backward pass
                            self.scaler.scale(loss * micro_weight).backward()

                            with torch.no_grad():
                                mb_pg_loss += pg_loss.item() * micro_weight
                                mb_v_loss += v_loss.item() * micro_weight
                                mb_entropy += entropy_loss.item() * micro_weight
                                mb_approx_kl += ((ratio - 1) - log_ratio).mean().item() * micro_weight
                                mb_clipfrac += (
                                    ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                                    * micro_weight
                                )

                        # Unscale gradients and clip
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        break
                    except oom_errors as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        self.optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if microbatch_size <= 1:
                            raise
                        microbatch_size = max(1, microbatch_size // 2)
                        print(
                            f"[ppo] CUDA OOM in update; reducing microbatch to {microbatch_size}"
                        )

                total_pg_loss += mb_pg_loss
                total_v_loss += mb_v_loss
                total_entropy += mb_entropy
                total_clipfrac += mb_clipfrac
                total_approx_kl += mb_approx_kl
                n_updates += 1

        metrics = {
            "loss/policy": total_pg_loss / max(n_updates, 1),
            "loss/value": total_v_loss / max(n_updates, 1),
            "loss/entropy": total_entropy / max(n_updates, 1),
            "ppo/clipfrac": total_clipfrac / max(n_updates, 1),
            "ppo/approx_kl": total_approx_kl / max(n_updates, 1),
            "ppo/entropy_coef": entropy_coef,  # NEW: log current entropy coef
        }

        return metrics

    def update_lr(self, progress: float):
        lr = self.cfg.lr * (1.0 - progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr