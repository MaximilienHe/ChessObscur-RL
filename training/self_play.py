"""
self_play.py — Self-play rollout collection.

CHANGES v8:
- League training support: some envs play against past checkpoints.
  When it's the opponent's turn in a league env, the opponent network picks
  the action. The learning network still evaluates all states for PPO.

CHANGES v5:
- Removed all parry/enemy_capture stats (illegal move removed)
- Parry stats now track only 3 outcomes: skip, good_move, self_capture
"""
import torch
import time
from typing import Dict, Tuple, Optional

from env.chess_obscur_env import ChessObscurEnv, ACTION_ACCEPT_LOSS
from model.network import ChessObscurNetwork
from config import Config
from utils.bitpack import pack_action_mask, packed_num_bytes


class RolloutBuffer:
    """Stores rollout data on GPU."""

    def __init__(self, T: int, N: int, obs_shape: Tuple, num_actions: int, device: str):
        self.T = T
        self.N = N
        self.num_actions = num_actions
        self.packed_mask_bytes = packed_num_bytes(num_actions)
        self.device = torch.device(device)

        self.obs = torch.zeros(T, N, *obs_shape, device=self.device)
        self.actions = torch.zeros(T, N, dtype=torch.int64, device=self.device)
        self.log_probs = torch.zeros(T, N, device=self.device)
        self.rewards = torch.zeros(T, N, device=self.device)
        self.dones = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.values = torch.zeros(T, N, device=self.device)
        self.legal_masks_packed = torch.zeros(
            T, N, self.packed_mask_bytes, dtype=torch.uint8, device=self.device
        )

        self.step = 0

    def insert(self, obs, actions, log_probs, rewards, dones, values, legal_masks):
        t = self.step
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.legal_masks_packed[t] = pack_action_mask(legal_masks)
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
            "legal_masks_packed": self.legal_masks_packed,
            "next_obs": next_obs,
        }


def collect_rollout(env: ChessObscurEnv, network: ChessObscurNetwork,
                    buffer: RolloutBuffer, obs: torch.Tensor,
                    use_amp: bool = False,
                    opponent_net: Optional[ChessObscurNetwork] = None,
                    league_mask: Optional[torch.Tensor] = None,
                    ) -> Tuple[torch.Tensor, Dict]:
    """
    Collect T steps of self-play experience.

    v8 league training: if opponent_net and league_mask are provided,
    envs where league_mask=True will use opponent_net when it's the
    opponent's turn. The learning network still evaluates all states
    for PPO (log_probs and values always come from the main network).

    Returns: (next_obs, stats_dict)
    """
    T = buffer.T
    games_completed = 0
    total_rewards = torch.zeros((), device=obs.device)
    white_wins = 0
    black_wins = 0
    draws = 0

    white_reward_sum = torch.zeros((), device=obs.device)
    black_reward_sum = torch.zeros((), device=obs.device)
    white_reward_count = torch.zeros((), dtype=torch.int64, device=obs.device)
    black_reward_count = torch.zeros((), dtype=torch.int64, device=obs.device)

    phase_counts = torch.zeros(3, dtype=torch.int64, device=obs.device)
    total_legal_actions = torch.zeros((), dtype=torch.int64, device=obs.device)
    game_lengths = []

    # v8: league win tracking
    league_games = 0
    league_wins = 0

    use_league = (opponent_net is not None and league_mask is not None
                  and league_mask.any())

    network.eval()
    with torch.no_grad():
        for t in range(T):
            legal_mask = env.get_legal_mask()

            phase_counts[0] += (env.phase == 0).sum()
            phase_counts[1] += (env.phase == 1).sum()
            phase_counts[2] += (env.phase == 2).sum()
            total_legal_actions += legal_mask.sum()

            no_legal = ~legal_mask.any(dim=1)
            if no_legal.any():
                legal_mask[no_legal, ACTION_ACCEPT_LOSS] = True

            # Main network evaluates ALL envs (for PPO log_probs and values)
            with torch.amp.autocast('cuda', enabled=use_amp):
                policy_logits, value = network(obs, legal_mask)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()

            # v8 league: override actions for league envs when it's the opponent's turn
            if use_league:
                # Opponent's turn = when current active player is NOT the agent
                is_opponent_turn = env.turn_is_white != env.agent_is_white
                opp_envs = league_mask & is_opponent_turn

                if opp_envs.any():
                    opp_idx = opp_envs.nonzero(as_tuple=True)[0]
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        opp_policy_logits, _ = opponent_net(obs[opp_idx], legal_mask[opp_idx])
                        opp_dist = torch.distributions.Categorical(logits=opp_policy_logits)
                        action[opp_idx] = opp_dist.sample()

            log_prob = dist.log_prob(action)
            value = value.squeeze(-1)

            next_obs, reward, done, info = env.step(action)

            buffer.insert(obs, action, log_prob, reward, done, value, legal_mask)

            agent_w = env.agent_is_white
            white_mask = agent_w
            black_mask = ~agent_w

            white_reward_sum += reward[white_mask].sum()
            black_reward_sum += reward[black_mask].sum()
            white_reward_count += white_mask.sum()
            black_reward_count += black_mask.sum()

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

                # v8: track league-specific win rate
                if use_league:
                    done_league = done & league_mask
                    if done_league.any():
                        league_games += done_league.sum().item()
                        league_results = info["result"][done_league]
                        league_agent_white = env.agent_is_white[done_league]
                        agent_wins = ((league_results == 1) & league_agent_white) | \
                                     ((league_results == 2) & ~league_agent_white)
                        league_wins += agent_wins.sum().item()

            total_rewards += reward.sum()
            obs = next_obs

    network.train()

    total_steps = T * env.N
    move_phases_seen = phase_counts[0].item()
    defense_phases_seen = phase_counts[1].item()
    parry_phases_seen = phase_counts[2].item()
    stats = {
        "rollout/games_completed": games_completed,
        "rollout/mean_reward": total_rewards.item() / total_steps,
        "rollout/white_wins": white_wins,
        "rollout/black_wins": black_wins,
        "rollout/draws": draws,
        "rollout/phase_move_frac": move_phases_seen / total_steps,
        "rollout/phase_defense_frac": defense_phases_seen / total_steps,
        "rollout/phase_parry_frac": parry_phases_seen / total_steps,
        "rollout/avg_legal_actions": total_legal_actions.item() / total_steps,
        "rollout/white_mean_reward": white_reward_sum.item() / max(white_reward_count.item(), 1),
        "rollout/black_mean_reward": black_reward_sum.item() / max(black_reward_count.item(), 1),
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

    # v8: league stats
    if league_games > 0:
        stats["league/games"] = league_games
        stats["league/win_rate"] = league_wins / league_games
        stats["league/pool_size"] = 0  # filled in by train.py

    # ── Parry stats: 3 outcomes only (skip, good_move, self_capture) ──
    parry_stats = env.get_and_reset_parry_stats()
    stats.update(parry_stats)

    pt = parry_stats["parry/total"]
    if pt > 0:
        stats["parry/self_capture_rate"] = parry_stats["parry/self_capture"] / pt
        stats["parry/good_move_rate"] = parry_stats["parry/good_move"] / pt
        stats["parry/skip_rate"] = parry_stats["parry/skip"] / pt

    # ── Capture quality stats ──
    capture_stats = env.get_and_reset_capture_stats()
    stats.update(capture_stats)
    ct = capture_stats["capture/total"]
    if ct > 0:
        stats["capture/high_attacker_rate"] = capture_stats["capture/high_attacker"] / ct

    # ── Check escape stats ──
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
