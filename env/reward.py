"""
reward.py — Reward shaping for Chess Obscur.

CHANGES v3 (parry fix + tuning):
- REWARD_PARRY_SELF_CAPTURE: increased penalty -0.08 -> -0.20 (scaled by piece value)
- REWARD_PARRY_SKIP: increased 0.005 -> 0.025 (make safe choice more attractive)
- REWARD_PARRY_MOVE_GOOD: increased 0.03 -> 0.08 (incentivize good parry moves)
- REWARD_PARRY_ENEMY_CAPTURE: NEW +0.05 (bonus for triggering defense on opponent's piece during parry)
- REWARD_STEP_PENALTY: reduced -0.003 -> -0.002 (less penalty, agent was playing too fast/recklessly)
- REWARD_CHECK_GIVEN: increased 0.08 -> 0.10 (stronger signal for check)
"""
import torch
from env.move_tables import EMPTY


# ── Terminal rewards ──
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = -0.3

# ── Intermediate shaping ──
REWARD_CAPTURE_SCALE = 0.10          # * piece_value of captured piece (positive for capturer)
REWARD_LOSE_PIECE_SCALE = -0.10      # * piece_value of own piece lost (negative for loser)
REWARD_CHECK_GIVEN = 0.10            # CHANGED: 0.08 -> 0.10, giving check is important
REWARD_BLOCK_SUCCESS = 0.06          # successfully blocked a capture
REWARD_PARRY_SUCCESS = 0.12          # parry is harder, reward more
REWARD_DEFENSE_FAIL = -0.01          # tried to defend but failed
REWARD_ACCEPT_LOSS = -0.02           # accepted loss without trying
REWARD_PARRY_MOVE_GOOD = 0.08        # CHANGED: 0.03 -> 0.08, good parry moves are very valuable
REWARD_PARRY_SELF_CAPTURE = -0.20    # CHANGED: -0.08 -> -0.20, HARSH penalty for eating your own piece (* piece_value)
REWARD_PARRY_ENEMY_CAPTURE = 0.05    # NEW: bonus for triggering defense on opponent piece during parry
REWARD_PARRY_SKIP = 0.025            # CHANGED: 0.005 -> 0.025, skipping is a valid safe choice
REWARD_CHECK_ATTEMPT_PENALTY = -0.05 # each wasted check attempt (3-check rule)
REWARD_STEP_PENALTY = -0.002         # CHANGED: -0.003 -> -0.002, less aggressive time pressure


def compute_material(board: torch.Tensor, piece_values: torch.Tensor, 
                     color_is_white: torch.Tensor) -> torch.Tensor:
    """
    Compute material balance for the active player.
    board: (N, 64) int8
    piece_values: (6,) float — indexed by piece_type (0-5)
    color_is_white: (N,) bool
    Returns: (N,) float — positive = active player has more material
    """
    N = board.shape[0]
    device = board.device
    
    white_mat = torch.zeros(N, device=device)
    black_mat = torch.zeros(N, device=device)
    
    for pt in range(6):
        w_piece = pt + 1   # 1..6
        b_piece = pt + 7   # 7..12
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
    """
    result_code: (N,) int
        0 = ongoing, 1 = white_win, 2 = black_win, 3 = draw
    active_is_white: (N,) bool — which color the agent was playing
    board: (N, 64) int8 — optional, for material-based draw rewards
    piece_values: (6,) float — optional, piece values for material calculation
    full_move_count: (N,) int — optional, to detect timeout draws
    max_steps: int — max steps before timeout draw (default: 150)
    Returns: (N,) float rewards
    """
    reward = torch.zeros_like(result_code, dtype=torch.float32)

    white_wins = result_code == 1
    black_wins = result_code == 2
    draws = result_code == 3

    # Agent played white
    reward = torch.where(white_wins & active_is_white,
                         torch.tensor(REWARD_WIN, device=reward.device), reward)
    reward = torch.where(black_wins & active_is_white,
                         torch.tensor(REWARD_LOSE, device=reward.device), reward)
    # Agent played black
    reward = torch.where(black_wins & ~active_is_white,
                         torch.tensor(REWARD_WIN, device=reward.device), reward)
    reward = torch.where(white_wins & ~active_is_white,
                         torch.tensor(REWARD_LOSE, device=reward.device), reward)

    # Draws
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
            base_draw_reward = -0.3
            material_scale = 0.2
            normalized_advantage = torch.tanh(material_advantage / 10.0)
            timeout_reward = base_draw_reward + material_scale * normalized_advantage
            reward = torch.where(timeout_draws, timeout_reward, reward)
        else:
            reward = torch.where(timeout_draws, torch.tensor(REWARD_DRAW, device=reward.device), reward)

    return reward