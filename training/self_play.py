"""
self_play.py — Self-play rollout collection.

CHANGES v4:
- Collect richer parry diagnostics (could_move, could_capture, skip_when_could_*)
- Collect capture quality stats (high_attacker usage)
- Collect check escape stats (escape by move vs capture, 3rd attempt behavior)
- Compute rates for all new metrics
"""
import torch
import time
from typing import Dict, Tuple

from env.chess_obscur_env import ChessObscurEnv
from model.network import ChessObscurNetwork
from config import Config


class RolloutBuffer:
    """Stores rollout data on GPU."""

    def __init__(self, T: int, N: int, obs_shape: Tuple, num_actions: int, device: str):
        self.T = T
        self.N = N
        self.device = torch.device(device)

        self.obs = torch.zeros(T, N, *obs_shape, device=self.device)
        self.actions = torch.zeros(T, N, dtype=torch.int64, device=self.device)
        self.log_probs = torch.zeros(T, N, device=self.device)
        self.rewards = torch.zeros(T, N, device=self.device)
        self.dones = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.values = torch.zeros(T, N, device=self.device)
        self.legal_masks = torch.zeros(T, N, num_actions, dtype=torch.bool, device=self.device)

        self.step = 0

    def insert(self, obs, actions, log_probs, rewards, dones, values, legal_masks):
        t = self.step
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.legal_masks[t] = legal_masks
        self.step += 1

    def get(self, next_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.step = 0
        return {
            "obs": self.obs,
            "actions": self.actions,
            "log_probs": self.log_probs,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "legal_masks": self.legal_masks,
            "next_obs": next_obs,
        }


def collect_rollout(env: ChessObscurEnv, network: ChessObscurNetwork,
                    buffer: RolloutBuffer, obs: torch.Tensor,
                    use_amp: bool = False) -> Tuple[torch.Tensor, Dict]:
    """
    Collect T steps of self-play experience.
    Returns: (next_obs, stats_dict)
    """
    T = buffer.T
    games_completed = 0
    total_rewards = 0.0
    white_wins = 0
    black_wins = 0
    draws = 0

    white_reward_sum = 0.0
    black_reward_sum = 0.0
    white_reward_count = 0
    black_reward_count = 0

    defense_phases_seen = 0
    parry_phases_seen = 0
    move_phases_seen = 0
    total_legal_actions = 0
    game_lengths = []

    network.eval()
    with torch.no_grad():
        for t in range(T):
            legal_mask = env.get_legal_mask()

            move_phases_seen += (env.phase == 0).sum().item()
            defense_phases_seen += (env.phase == 1).sum().item()
            parry_phases_seen += (env.phase == 2).sum().item()
            total_legal_actions += legal_mask.sum().item()

            no_legal = ~legal_mask.any(dim=1)
            if no_legal.any():
                legal_mask[no_legal, 4162] = True

            with torch.amp.autocast('cuda', enabled=use_amp):
                action, log_prob, entropy, value = network.get_action_and_value(obs, legal_mask)

            next_obs, reward, done, info = env.step(action)

            buffer.insert(obs, action, log_prob, reward, done, value, legal_mask)

            agent_w = env.agent_is_white
            white_mask = agent_w
            black_mask = ~agent_w

            white_reward_sum += reward[white_mask].sum().item()
            black_reward_sum += reward[black_mask].sum().item()
            white_reward_count += white_mask.sum().item()
            black_reward_count += black_mask.sum().item()

            if done.any():
                n_done = done.sum().item()
                games_completed += n_done
                results = info["result"][done]
                white_wins += (results == 1).sum().item()
                black_wins += (results == 2).sum().item()
                draws += (results == 3).sum().item()

                move_counts = info["full_move_count"][done]
                for mc in move_counts:
                    game_lengths.append(mc.item())

            total_rewards += reward.sum().item()
            obs = next_obs

    network.train()

    total_steps = T * env.N
    stats = {
        "rollout/games_completed": games_completed,
        "rollout/mean_reward": total_rewards / total_steps,
        "rollout/white_wins": white_wins,
        "rollout/black_wins": black_wins,
        "rollout/draws": draws,
        "rollout/phase_move_frac": move_phases_seen / total_steps,
        "rollout/phase_defense_frac": defense_phases_seen / total_steps,
        "rollout/phase_parry_frac": parry_phases_seen / total_steps,
        "rollout/avg_legal_actions": total_legal_actions / total_steps,
        "rollout/white_mean_reward": white_reward_sum / max(white_reward_count, 1),
        "rollout/black_mean_reward": black_reward_sum / max(black_reward_count, 1),
    }

    if games_completed > 0:
        stats["game/win_rate"] = (white_wins + black_wins) / games_completed
        stats["game/draw_rate"] = draws / games_completed
        stats["game/white_win_rate"] = white_wins / games_completed
        stats["game/black_win_rate"] = black_wins / games_completed
    if game_lengths:
        stats["game/avg_length"] = sum(game_lengths) / len(game_lengths)
        stats["game/max_length"] = max(game_lengths)
        stats["game/min_length"] = min(game_lengths)

    # ── Parry outcome stats ──
    parry_stats = env.get_and_reset_parry_stats()
    stats.update(parry_stats)

    pt = parry_stats["parry/total"]
    if pt > 0:
        stats["parry/self_capture_rate"] = parry_stats["parry/self_capture"] / pt
        stats["parry/good_move_rate"] = parry_stats["parry/good_move"] / pt
        stats["parry/skip_rate"] = parry_stats["parry/skip"] / pt
        stats["parry/enemy_capture_rate"] = parry_stats["parry/enemy_capture"] / pt

        # NEW v4: diagnostic rates
        stats["parry/could_move_rate"] = parry_stats["parry/could_move"] / pt
        stats["parry/could_enemy_capture_rate"] = parry_stats["parry/could_enemy_capture"] / pt

        skip_total = parry_stats["parry/skip"]
        if skip_total > 0:
            stats["parry/skip_when_could_move_rate"] = parry_stats["parry/skip_when_could_move"] / skip_total
            stats["parry/skip_when_could_capture_rate"] = parry_stats["parry/skip_when_could_capture"] / skip_total

    # ── NEW v4: Capture quality stats ──
    capture_stats = env.get_and_reset_capture_stats()
    stats.update(capture_stats)
    ct = capture_stats["capture/total"]
    if ct > 0:
        stats["capture/high_attacker_rate"] = capture_stats["capture/high_attacker"] / ct

    # ── NEW v4: Check escape stats ──
    check_stats = env.get_and_reset_check_stats()
    stats.update(check_stats)
    total_check_escapes = check_stats["check/escape_by_move"] + check_stats["check/escape_by_capture"]
    if total_check_escapes > 0:
        stats["check/escape_by_move_rate"] = check_stats["check/escape_by_move"] / total_check_escapes
        stats["check/escape_by_capture_rate"] = check_stats["check/escape_by_capture"] / total_check_escapes
    total_3rd = check_stats["check/3rd_attempt_capture"] + check_stats["check/3rd_attempt_move"]
    if total_3rd > 0:
        stats["check/3rd_attempt_move_rate"] = check_stats["check/3rd_attempt_move"] / total_3rd

    return obs, stats