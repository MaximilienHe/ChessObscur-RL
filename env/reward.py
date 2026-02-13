"""
reward.py — Reward shaping for Chess Obscur.

CHANGES v5 (parry fix: remove illegal enemy_capture):
- REMOVED: REWARD_PARRY_ENEMY_CAPTURE (this was never a legal move — parry can't capture enemy pieces)
- Parry now only has 3 outcomes: skip, good_move (empty square), self_capture (own piece)
"""
import torch
from env.move_tables import EMPTY


# ── Terminal rewards ──
REWARD_WIN = 1.5
REWARD_LOSE = -1.0
REWARD_DRAW = -0.8

BASE_DRAW_REWARD = -0.9
MATERIAL_SCALE = 0.15

# ── Intermediate shaping ──
REWARD_CAPTURE_SCALE = 0.25
REWARD_LOSE_PIECE_SCALE = -0.06

# ── Check escape shaping ──
REWARD_CHECK_ESCAPE_MOVE = 0.03
REWARD_CHECK_3RD_CAPTURE_PENALTY = -0.08
REWARD_CHECK_GIVEN = 0.15
REWARD_CHECK_2ND_ATTEMPT = 0.25
REWARD_CHECK_ATTEMPT_PENALTY = -0.08

REWARD_BLOCK_SUCCESS = 0.05
REWARD_DEFENSE_FAIL = -0.01
REWARD_ACCEPT_LOSS = -0.04

REWARD_PARRY_SUCCESS = 0.18
REWARD_PARRY_MOVE_GOOD = 0.15
REWARD_PARRY_SELF_CAPTURE = -0.30       # harsh penalty (* piece_value)
REWARD_PARRY_SKIP = -0.03

REWARD_STEP_PENALTY = -0.005

# ── Capture quality bonus ──
REWARD_CAPTURE_ATTACKER_BONUS = 0.02



def compute_material(board: torch.Tensor, piece_values: torch.Tensor, 
                     color_is_white: torch.Tensor) -> torch.Tensor:
    N = board.shape[0]
    device = board.device
    
    white_mat = torch.zeros(N, device=device)
    black_mat = torch.zeros(N, device=device)
    
    for pt in range(6):
        w_piece = pt + 1
        b_piece = pt + 7
        w_count = (board == w_piece).sum(dim=1).float()
        b_count = (board == b_piece).sum(dim=1).float()
        white_mat += w_count * piece_values[pt]
        black_mat += b_count * piece_values[pt]
    
    my_mat = torch.where(color_is_white, white_mat, black_mat)
    opp_mat = torch.where(color_is_white, black_mat, white_mat)
    return my_mat - opp_mat


def reward_terminal(result_code: torch.Tensor, active_is_white: torch.Tensor,
                    board: torch.Tensor = None, piece_values: torch.Tensor = None,
                    full_move_count: torch.Tensor = None, max_steps: int = 150) -> torch.Tensor:
    reward = torch.zeros_like(result_code, dtype=torch.float32)

    white_wins = result_code == 1
    black_wins = result_code == 2
    draws = result_code == 3

    reward = torch.where(white_wins & active_is_white,
                         torch.tensor(REWARD_WIN, device=reward.device), reward)
    reward = torch.where(black_wins & active_is_white,
                         torch.tensor(REWARD_LOSE, device=reward.device), reward)
    reward = torch.where(black_wins & ~active_is_white,
                         torch.tensor(REWARD_WIN, device=reward.device), reward)
    reward = torch.where(white_wins & ~active_is_white,
                         torch.tensor(REWARD_LOSE, device=reward.device), reward)

    if draws.any():
        timeout_draws = draws.clone()
        if full_move_count is not None:
            timeout_draws = draws & (full_move_count >= max_steps * 2)
        else:
            timeout_draws = torch.zeros_like(draws)

        other_draws = draws & ~timeout_draws
        reward = torch.where(other_draws, torch.tensor(REWARD_DRAW, device=reward.device), reward)

        if timeout_draws.any() and board is not None and piece_values is not None:
            material_advantage = compute_material(board, piece_values, active_is_white)
            base_draw_reward = BASE_DRAW_REWARD
            material_scale = MATERIAL_SCALE
            normalized_advantage = torch.tanh(material_advantage / 10.0)
            timeout_reward = base_draw_reward + material_scale * normalized_advantage
            reward = torch.where(timeout_draws, timeout_reward, reward)
        else:
            reward = torch.where(timeout_draws, torch.tensor(REWARD_DRAW, device=reward.device), reward)

    return reward