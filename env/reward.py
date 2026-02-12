"""
reward.py — Reward shaping for Chess Obscur.

CHANGES v4 (aggression + parry activation + draw reduction + smart check):
- REWARD_CAPTURE_SCALE: 0.14 -> 0.20 (agent is not aggressive enough, needs more capture incentive)
- REWARD_LOSE_PIECE_SCALE: -0.08 -> -0.06 (agent was too cautious, slightly softer loss penalty)
- REWARD_PARRY_MOVE_GOOD: 0.05 -> 0.15 (KEY: parry good_move was 17% dropping to 9%, need strong incentive)
- REWARD_PARRY_SKIP: 0.00 -> -0.03 (KEY: skip was 91%! must penalize skipping parry)
- REWARD_PARRY_ENEMY_CAPTURE: 0.12 -> 0.25 (was 0.0% usage — huge bonus for using parry aggressively)
- REWARD_PARRY_SELF_CAPTURE: -0.20 -> -0.30 (keep harsh for self-capture, scaled by piece value)
- REWARD_PARRY_SUCCESS: 0.12 -> 0.18 (increase reward for choosing parry in defense, it enables good parry moves)
- REWARD_BLOCK_SUCCESS: 0.06 -> 0.05 (slightly nerf block relative to parry)
- REWARD_DRAW: -0.3 -> -0.5 (KEY: draw rate was 75%!! stronger draw penalty)
- REWARD_STEP_PENALTY: -0.002 -> -0.003 (slightly more time pressure, games averaged 364 half-moves)
- NEW: REWARD_CAPTURE_HIGH_ATTACKER: bonus for using high-attack pieces for captures
- NEW: REWARD_CHECK_ESCAPE_SAFE: bonus for escaping check by moving (not just capturing)
- NEW: REWARD_CHECK_ESCAPE_GREEDY_3RD: penalty for attempting capture on 3rd check (should play safe)
"""
import torch
from env.move_tables import EMPTY


# ── Terminal rewards ──
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = -0.5              # CHANGED: -0.3 -> -0.5, draw rate was 75%!

# ── Intermediate shaping ──
REWARD_CAPTURE_SCALE = 0.20          # CHANGED: 0.14 -> 0.20, boost aggression
REWARD_LOSE_PIECE_SCALE = -0.06      # CHANGED: -0.08 -> -0.06, less cautious
REWARD_CHECK_GIVEN = 0.06            # keep same
REWARD_BLOCK_SUCCESS = 0.05          # CHANGED: 0.06 -> 0.05, slight nerf vs parry
REWARD_PARRY_SUCCESS = 0.18          # CHANGED: 0.12 -> 0.18, encourage choosing parry in defense
REWARD_DEFENSE_FAIL = -0.01          # tried to defend but failed
REWARD_ACCEPT_LOSS = -0.04           # CHANGED: -0.02 -> -0.04, penalize passive acceptance more
REWARD_PARRY_MOVE_GOOD = 0.15        # CHANGED: 0.05 -> 0.15, KEY: good_move rate was dropping to 9%
REWARD_PARRY_SELF_CAPTURE = -0.30    # CHANGED: -0.20 -> -0.30, harsh penalty (* piece_value)
REWARD_PARRY_ENEMY_CAPTURE = 0.25    # CHANGED: 0.12 -> 0.25, enemy_capture rate was 0.0%!
REWARD_PARRY_SKIP = -0.03            # CHANGED: 0.00 -> -0.03, skip rate was 91%, must penalize
REWARD_CHECK_ATTEMPT_PENALTY = -0.05 # keep same
REWARD_STEP_PENALTY = -0.003         # CHANGED: -0.002 -> -0.003, avg game length was 364

# ── NEW: capture quality bonus (use high-attack piece) ──
REWARD_CAPTURE_ATTACKER_BONUS = 0.02  # per point of attacker attack_stat above defender

# ── NEW: check escape shaping ──
REWARD_CHECK_ESCAPE_MOVE = 0.03      # bonus for escaping check by moving (not capture)
REWARD_CHECK_3RD_CAPTURE_PENALTY = -0.08  # penalty for trying capture on 3rd check attempt


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

    # Draws — stronger penalty, especially for timeout draws
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
            # Timeout draws are WORSE than regular draws — agent should have won
            base_draw_reward = -0.6  # CHANGED: -0.3 -> -0.6 for timeout
            material_scale = 0.25    # CHANGED: 0.2 -> 0.25
            normalized_advantage = torch.tanh(material_advantage / 10.0)
            timeout_reward = base_draw_reward + material_scale * normalized_advantage
            reward = torch.where(timeout_draws, timeout_reward, reward)
        else:
            reward = torch.where(timeout_draws, torch.tensor(REWARD_DRAW, device=reward.device), reward)

    return reward