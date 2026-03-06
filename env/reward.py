"""
reward.py — Reward shaping for Chess Obscur.

CHANGES v8 (progressive draw penalty + parry rebalance — fresh start):
- REWARD_DRAW: flat -1.3 -> progressive function of game duration.
  Early draws (stalemate) = -0.5, late draws (timeout/50-move) = -2.0.
  This incentivizes the agent to play for wins in shorter games instead of
  settling into the draw Nash equilibrium (79% draws at v7's 493M checkpoint).
- REWARD_PARRY_MOVE_GOOD: 0.15 -> 0.30 — parry good_move_rate dropped from 46% to 30%
  while skip_rate rose to 69%. Stronger incentive to exploit parry offensively.
- REWARD_PARRY_SKIP: -0.03 -> -0.08 — make skip less of a free action.

CHANGES v7 (draw penalty + parry self-capture fix — on top of v6 checkpoint at 122M steps):
- REWARD_WIN: 1.5 -> 2.0 — win signal trop faible face au Nash equilibrium draw (77%).
- REWARD_DRAW: -0.8 -> -1.3 — pénaliser les nuls plus fort pour casser le Nash.
- BASE_DRAW_REWARD: -0.9 -> -1.3 — cohérence avec REWARD_DRAW pour les timeout draws.
- REWARD_PARRY_SELF_CAPTURE: -0.80 -> -1.50 — régression 2.7%->9.5%.

CHANGES v6 (check escape + parry self-capture fix):
- REMOVED: REWARD_CHECK_3RD_CAPTURE_PENALTY — penalisait les captures valides.
- ADDED: REWARD_CHECK_ESCAPE_SUCCESS — outcome-based, scales with urgency.
- REWARD_PARRY_SELF_CAPTURE: -0.30 -> -0.80.
"""
import torch
from env.move_tables import EMPTY


# ── Terminal rewards ──
REWARD_WIN = 2.0
REWARD_LOSE = -1.0

# ── Progressive draw penalty v8 ──
# Instead of a flat REWARD_DRAW, the penalty scales with game duration.
# Early draws (stalemate, repetition) are less punished than late timeout draws.
REWARD_DRAW_EARLY = -0.5       # v8: draw before 30% of max_steps
REWARD_DRAW_LATE = -2.0        # v8: draw at/near timeout
MATERIAL_SCALE = 0.15

# ── Intermediate shaping ──
REWARD_CAPTURE_SCALE = 0.25
REWARD_LOSE_PIECE_SCALE = -0.06

# ── Check escape shaping ──
REWARD_CHECK_ESCAPE_SUCCESS = 0.06
REWARD_CHECK_GIVEN = 0.15
REWARD_CHECK_2ND_ATTEMPT = 0.25
REWARD_CHECK_ATTEMPT_PENALTY = -0.08

REWARD_BLOCK_SUCCESS = 0.05
REWARD_DEFENSE_FAIL = -0.01
REWARD_ACCEPT_LOSS = -0.04

REWARD_PARRY_SUCCESS = 0.18
REWARD_PARRY_MOVE_GOOD = 0.30    # v8: 0.15 -> 0.30, incentivize offensive parry
REWARD_PARRY_SELF_CAPTURE = -1.50
REWARD_PARRY_SKIP = -0.08        # v8: -0.03 -> -0.08, skip is not free

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


def _progressive_draw_penalty(full_move_count: torch.Tensor, max_steps: int) -> torch.Tensor:
    """Compute progressive draw penalty based on game duration.

    Short games that draw (stalemate, repetition) get a milder penalty.
    Long games that draw (timeout, 50-move rule) get a harsher penalty.
    This incentivizes playing for wins instead of settling into draws.
    """
    # progress: 0.0 = game just started, 1.0 = reached max_steps
    progress = (full_move_count.float() / (max_steps * 2)).clamp(0.0, 1.0)
    # Lerp between early and late draw penalties
    return REWARD_DRAW_EARLY + (REWARD_DRAW_LATE - REWARD_DRAW_EARLY) * progress


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
        # v8: progressive draw penalty based on game duration
        if full_move_count is not None:
            draw_penalty = _progressive_draw_penalty(full_move_count, max_steps)

            # For timeout draws, also factor in material advantage
            timeout_draws = draws & (full_move_count >= max_steps * 2)
            other_draws = draws & ~timeout_draws

            reward = torch.where(other_draws, draw_penalty, reward)

            if timeout_draws.any() and board is not None and piece_values is not None:
                material_advantage = compute_material(board, piece_values, active_is_white)
                normalized_advantage = torch.tanh(material_advantage / 10.0)
                timeout_reward = draw_penalty + MATERIAL_SCALE * normalized_advantage
                reward = torch.where(timeout_draws, timeout_reward, reward)
            else:
                reward = torch.where(timeout_draws, draw_penalty, reward)
        else:
            # Fallback: use late penalty (worst case)
            reward = torch.where(draws, torch.tensor(REWARD_DRAW_LATE, device=reward.device), reward)

    return reward