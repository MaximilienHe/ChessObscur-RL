"""
chess_obscur_env.py — Fully vectorized GPU Chess Obscur environment.

CHANGES v9 (performance + correctness overhaul):
  PERF: Vectorized _resolve_defense, _apply_board_moves, _enforce_check —
        eliminated all per-env Python loops and .item() GPU syncs in step().
  PERF: Vectorized parry legal mask generation in get_legal_mask().
  PERF: Optimized _build_obs with batched piece plane computation.
  FIX:  Castling through check now properly blocked (transit squares checked).
  FIX:  Removed dead underpromotion action slots (4163 → 4099 actions).
  NEW:  Zobrist hashing + threefold repetition detection.
  NEW:  Symmetric rewards (win/loss), draws never worse than loss.

CHANGES v5 (parry rules fix):
  FIX: Removed illegal "enemy capture" during parry phase.
"""
import torch
from typing import Tuple, Optional, Dict

from env.move_tables import (
    MoveTables, EMPTY,
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
)
from env.reward import (
    reward_terminal,
    REWARD_CAPTURE_SCALE, REWARD_LOSE_PIECE_SCALE,
    REWARD_BLOCK_SUCCESS, REWARD_PARRY_SUCCESS,
    REWARD_DEFENSE_FAIL, REWARD_ACCEPT_LOSS, REWARD_PARRY_MOVE_GOOD,
    REWARD_PARRY_SKIP, REWARD_PARRY_SELF_CAPTURE,
    REWARD_STEP_PENALTY, REWARD_CHECK_ATTEMPT_PENALTY,
    REWARD_CAPTURE_ATTACKER_BONUS,
    REWARD_CHECK_ESCAPE_SUCCESS,
    REWARD_CHECK_GIVEN, REWARD_CHECK_2ND_ATTEMPT
)

PHASE_MOVE = 0
PHASE_DEFENSE = 1
PHASE_PARRY = 2
PHASE_FINISHED = 3
RESULT_ONGOING = 0
RESULT_WHITE_WIN = 1
RESULT_BLACK_WIN = 2
RESULT_DRAW = 3

# v9: defense action indices (no more underpromotion gap)
ACTION_ATTEMPT_BLOCK = 4096
ACTION_ATTEMPT_PARRY = 4097
ACTION_ACCEPT_LOSS = 4098

MAX_ZOBRIST_HISTORY = 400


class ChessObscurEnv:
    def __init__(self, num_envs: int, device: str = "cuda", max_steps: int = 150):
        self.N = num_envs
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.tables = MoveTables(device)
        self._precompute_attack_tables()
        self._precompute_zobrist()

        dev = self.device
        self.board = torch.zeros(num_envs, 64, dtype=torch.int8, device=dev)
        self.turn_is_white = torch.ones(num_envs, dtype=torch.bool, device=dev)
        self.phase = torch.zeros(num_envs, dtype=torch.int8, device=dev)
        self.result = torch.zeros(num_envs, dtype=torch.int8, device=dev)
        self.castling = torch.ones(num_envs, 4, dtype=torch.bool, device=dev)
        self.en_passant = torch.full((num_envs,), -1, dtype=torch.int16, device=dev)
        self.check_attempts = torch.zeros(num_envs, 2, dtype=torch.int8, device=dev)
        self.half_moves = torch.zeros(num_envs, dtype=torch.int16, device=dev)
        self.full_move_count = torch.zeros(num_envs, dtype=torch.int16, device=dev)
        self.pending_attacker_sq = torch.full((num_envs,), -1, dtype=torch.int16, device=dev)
        self.pending_target_sq = torch.full((num_envs,), -1, dtype=torch.int16, device=dev)
        self.pending_attacker_piece = torch.zeros(num_envs, dtype=torch.int8, device=dev)
        self.pending_defender_piece = torch.zeros(num_envs, dtype=torch.int8, device=dev)
        self.pending_attacker_color_white = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self.parry_square = torch.full((num_envs,), -1, dtype=torch.int16, device=dev)
        self.parry_controller_is_white = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self.agent_is_white = torch.ones(num_envs, dtype=torch.bool, device=dev)

        # v9: Zobrist history for threefold repetition
        self.zobrist_history = torch.zeros(num_envs, MAX_ZOBRIST_HISTORY,
                                           dtype=torch.int64, device=dev)
        self.zobrist_len = torch.zeros(num_envs, dtype=torch.int16, device=dev)

        # ── Parry outcome counters ──
        self.parry_self_capture_count = 0
        self.parry_good_move_count = 0
        self.parry_skip_count = 0
        self.parry_total_count = 0

        # ── Capture quality stats ──
        self.capture_total_count = 0
        self.capture_high_attacker_count = 0

        # ── Check escape stats ──
        self.check_escape_by_move_count = 0
        self.check_escape_by_capture_count = 0
        self.check_3rd_attempt_capture_count = 0
        self.check_3rd_attempt_move_count = 0

        self.reset()

    def _precompute_attack_tables(self):
        dev = self.device
        kt = self.tables.knight_moves
        self.knight_attack_table = torch.zeros(64, 64, dtype=torch.bool, device=dev)
        for sq in range(64):
            for j in range(8):
                t = kt[sq, j].item()
                if t >= 0: self.knight_attack_table[sq, t] = True

        km = self.tables.king_moves
        self.king_attack_table = torch.zeros(64, 64, dtype=torch.bool, device=dev)
        for sq in range(64):
            for j in range(8):
                t = km[sq, j].item()
                if t >= 0: self.king_attack_table[sq, t] = True

        self.w_pawn_attack_table = torch.zeros(64, 64, dtype=torch.bool, device=dev)
        self.b_pawn_attack_table = torch.zeros(64, 64, dtype=torch.bool, device=dev)
        for sq in range(64):
            f, r = sq % 8, sq // 8
            if r < 7:
                if f > 0: self.w_pawn_attack_table[sq, (f-1)+(r+1)*8] = True
                if f < 7: self.w_pawn_attack_table[sq, (f+1)+(r+1)*8] = True
            if r > 0:
                if f > 0: self.b_pawn_attack_table[sq, (f-1)+(r-1)*8] = True
                if f < 7: self.b_pawn_attack_table[sq, (f+1)+(r-1)*8] = True

        ray = self.tables.ray_moves
        diag_dirs = {1, 3, 5, 7}
        self.between_mask = torch.zeros(64, 64, 64, dtype=torch.bool, device=dev)
        self.ray_aligned = torch.zeros(64, 64, dtype=torch.bool, device=dev)
        self.ray_type = torch.zeros(64, 64, dtype=torch.int8, device=dev)

        for sq in range(64):
            for d in range(8):
                ray_sqs = []
                for step in range(7):
                    t = ray[sq, d, step].item()
                    if t < 0: break
                    ray_sqs.append(t)
                for idx_t, target in enumerate(ray_sqs):
                    self.ray_aligned[sq, target] = True
                    self.ray_type[sq, target] = 1 if d in diag_dirs else 2
                    for b in ray_sqs[:idx_t]:
                        self.between_mask[sq, target, b] = True

    def _precompute_zobrist(self):
        """Precompute Zobrist hash table for threefold repetition detection."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(314159265)

        # 13 piece types (0=empty, 1-12=pieces) x 64 squares
        self.zobrist_table = torch.randint(
            0, 2**62, (13, 64), dtype=torch.int64, generator=gen
        ).to(self.device)
        self.zobrist_turn_hash = torch.randint(
            0, 2**62, (1,), dtype=torch.int64, generator=gen
        ).to(self.device)
        self.zobrist_castling = torch.randint(
            0, 2**62, (4,), dtype=torch.int64, generator=gen
        ).to(self.device)
        self.zobrist_en_passant = torch.randint(
            0, 2**62, (64,), dtype=torch.int64, generator=gen
        ).to(self.device)
        self.zobrist_check_attempts = torch.randint(
            0, 2**62, (2, 4), dtype=torch.int64, generator=gen
        ).to(self.device)

    # ══════════════════════════════════════════════
    #  RESET
    # ══════════════════════════════════════════════

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.N, dtype=torch.bool, device=self.device)
        n = mask.sum().item()
        if n == 0: return self._build_obs()
        dev = self.device
        init = torch.zeros(n, 64, dtype=torch.int8, device=dev)
        bw = torch.tensor([W_ROOK,W_KNIGHT,W_BISHOP,W_QUEEN,W_KING,W_BISHOP,W_KNIGHT,W_ROOK], dtype=torch.int8, device=dev)
        bb = torch.tensor([B_ROOK,B_KNIGHT,B_BISHOP,B_QUEEN,B_KING,B_BISHOP,B_KNIGHT,B_ROOK], dtype=torch.int8, device=dev)
        init[:,0:8]=bw; init[:,8:16]=W_PAWN; init[:,48:56]=B_PAWN; init[:,56:64]=bb
        self.board[mask]=init; self.turn_is_white[mask]=True; self.phase[mask]=PHASE_MOVE
        self.result[mask]=RESULT_ONGOING; self.castling[mask]=True; self.en_passant[mask]=-1
        self.check_attempts[mask]=0; self.half_moves[mask]=0; self.full_move_count[mask]=0
        self.pending_attacker_sq[mask]=-1
        self.pending_target_sq[mask]=-1; self.parry_square[mask]=-1
        self.pending_attacker_piece[mask] = EMPTY
        self.pending_defender_piece[mask] = EMPTY
        self.pending_attacker_color_white[mask] = False
        self.parry_controller_is_white[mask] = False
        self.agent_is_white[mask] = torch.rand(n, device=dev) > 0.5
        # v9: reset zobrist history and record the starting position.
        self.zobrist_history[mask] = 0
        self.zobrist_len[mask] = 0
        self._update_zobrist(mask.nonzero(as_tuple=True)[0])
        return self._build_obs()

    def set_max_steps(self, new_max_steps: int):
        self.max_steps = new_max_steps

    def get_and_reset_parry_stats(self):
        stats = {
            "parry/total": self.parry_total_count,
            "parry/self_capture": self.parry_self_capture_count,
            "parry/good_move": self.parry_good_move_count,
            "parry/skip": self.parry_skip_count,
        }
        self.parry_total_count = 0
        self.parry_self_capture_count = 0
        self.parry_good_move_count = 0
        self.parry_skip_count = 0
        return stats

    def get_and_reset_capture_stats(self):
        stats = {
            "capture/total": self.capture_total_count,
            "capture/high_attacker": self.capture_high_attacker_count,
        }
        self.capture_total_count = 0
        self.capture_high_attacker_count = 0
        return stats

    def get_and_reset_check_stats(self):
        stats = {
            "check/escape_by_move": self.check_escape_by_move_count,
            "check/escape_by_capture": self.check_escape_by_capture_count,
            "check/3rd_attempt_capture": self.check_3rd_attempt_capture_count,
            "check/3rd_attempt_move": self.check_3rd_attempt_move_count,
        }
        self.check_escape_by_move_count = 0
        self.check_escape_by_capture_count = 0
        self.check_3rd_attempt_capture_count = 0
        self.check_3rd_attempt_move_count = 0
        return stats

    # ══════════════════════════════════════════════
    #  ZOBRIST HASHING (v9)
    # ══════════════════════════════════════════════

    def _compute_zobrist_hash(self, env_indices):
        """Compute Zobrist hash for given envs. Fully vectorized."""
        boards = self.board[env_indices]           # (M, 64)
        is_white = self.turn_is_white[env_indices]  # (M,)
        M = boards.shape[0]
        dev = self.device

        sq_idx = torch.arange(64, device=dev).unsqueeze(0).expand(M, -1)
        piece_idx = boards.long()  # (M, 64)
        values = self.zobrist_table[piece_idx, sq_idx]  # (M, 64)

        # XOR tree reduction (6 steps instead of 63)
        while values.shape[1] > 1:
            n = values.shape[1]
            if n % 2 == 1:
                last = values[:, -1:]
                values = values[:, :-1]
            else:
                last = None
            values = values[:, 0::2] ^ values[:, 1::2]
            if last is not None:
                values = torch.cat([values, last], dim=1)
        h = values.squeeze(1)

        # Include turn-to-move.
        h = h ^ torch.where(~is_white, self.zobrist_turn_hash.expand(M),
                            torch.zeros(M, dtype=torch.int64, device=dev))

        # Castling rights are part of the chess position identity.
        castling = self.castling[env_indices]
        zero = torch.zeros(M, dtype=torch.int64, device=dev)
        for i in range(4):
            h = h ^ torch.where(castling[:, i], self.zobrist_castling[i], zero)

        # En passant availability also affects legal moves.
        ep = self.en_passant[env_indices].long()
        has_ep = ep >= 0
        if has_ep.any():
            h[has_ep] = h[has_ep] ^ self.zobrist_en_passant[ep[has_ep]]

        # In Chess Obscur, the accumulated check count changes the future game state.
        ca = self.check_attempts[env_indices].long().clamp(min=0, max=3)
        h = h ^ self.zobrist_check_attempts[0, ca[:, 0]]
        h = h ^ self.zobrist_check_attempts[1, ca[:, 1]]
        return h

    def _update_zobrist(self, env_indices):
        """Record current position hash in history."""
        if env_indices.shape[0] == 0:
            return
        h = self._compute_zobrist_hash(env_indices)
        lens = self.zobrist_len[env_indices].long().clamp(max=MAX_ZOBRIST_HISTORY - 1)
        self.zobrist_history[env_indices, lens] = h
        self.zobrist_len[env_indices] = (lens + 1).clamp(max=MAX_ZOBRIST_HISTORY).short()

    def _check_threefold(self, env_indices):
        """Check if current position has appeared 3+ times. Returns bool mask."""
        if env_indices.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        h = self._compute_zobrist_hash(env_indices)
        M = env_indices.shape[0]
        lens = self.zobrist_len[env_indices].long()
        history = self.zobrist_history[env_indices]  # (M, MAX_ZOBRIST_HISTORY)
        valid = torch.arange(MAX_ZOBRIST_HISTORY, device=self.device).unsqueeze(0) < lens.unsqueeze(1)
        matches = (history == h.unsqueeze(1)) & valid
        count = matches.sum(dim=1)
        return count >= 3

    # ══════════════════════════════════════════════
    #  OBSERVATION (v9: batched piece planes)
    # ══════════════════════════════════════════════

    def _build_obs(self):
        N, dev = self.N, self.device
        obs = torch.zeros(N, 19, 8, 8, device=dev)
        board = self.board
        is_w = self.turn_is_white.view(N, 1, 1).float()

        # v9: batched piece plane computation (2 comparisons instead of 12)
        pieces = board.unsqueeze(1)  # (N, 1, 64)
        w_pieces = torch.arange(1, 7, device=dev).view(1, 6, 1)
        b_pieces = torch.arange(7, 13, device=dev).view(1, 6, 1)
        w_mask = (pieces == w_pieces).float().view(N, 6, 8, 8)
        b_mask = (pieces == b_pieces).float().view(N, 6, 8, 8)
        is_w_4d = is_w.unsqueeze(1)
        obs[:, :6] = w_mask * is_w_4d + b_mask * (1 - is_w_4d)
        obs[:, 6:12] = b_mask * is_w_4d + w_mask * (1 - is_w_4d)

        ep = self.en_passant; ev = ep >= 0
        if ev.any():
            ef = torch.zeros(N, 64, device=dev)
            ef.scatter_(1, ep.long().clamp(0, 63).unsqueeze(1), ev.float().unsqueeze(1))
            obs[:, 12] = ef.view(N, 8, 8)
        c = self.castling.float(); iw = self.turn_is_white.float()
        obs[:, 13, 0, :] = (c[:, 0]*iw+c[:, 2]*(1-iw)).unsqueeze(1).expand(-1, 8)
        obs[:, 13, 1, :] = (c[:, 1]*iw+c[:, 3]*(1-iw)).unsqueeze(1).expand(-1, 8)
        obs[:, 13, 2, :] = (c[:, 2]*iw+c[:, 0]*(1-iw)).unsqueeze(1).expand(-1, 8)
        obs[:, 13, 3, :] = (c[:, 3]*iw+c[:, 1]*(1-iw)).unsqueeze(1).expand(-1, 8)
        obs[:, 14] = self.turn_is_white.float().view(N, 1, 1).expand(-1, 8, 8)
        obs[:, 15] = self._is_in_check_batched(self.board, self.turn_is_white).float().view(N, 1, 1).expand(-1, 8, 8)
        obs[:, 16] = (self.phase.float()/3).view(N, 1, 1).expand(-1, 8, 8)
        ca = self.check_attempts
        obs[:, 17] = torch.where(self.turn_is_white, ca[:, 0], ca[:, 1]).float().div(3).view(N, 1, 1).expand(-1, 8, 8)
        obs[:, 18] = (self.half_moves.float()/100).view(N, 1, 1).expand(-1, 8, 8)
        return obs

    # ══════════════════════════════════════════════
    #  BATCHED ATTACK DETECTION
    # ══════════════════════════════════════════════

    def _is_sq_attacked_batched(self, boards, sq, by_white):
        M = boards.shape[0]; dev = boards.device
        result = torch.zeros(M, dtype=torch.bool, device=dev)
        sq_l = sq.long()

        kt_mask = self.knight_attack_table[sq_l]
        en_k = torch.where(by_white.unsqueeze(1), boards == W_KNIGHT, boards == B_KNIGHT)
        result |= (kt_mask & en_k).any(dim=1)

        kg_mask = self.king_attack_table[sq_l]
        en_kg = torch.where(by_white.unsqueeze(1), boards == W_KING, boards == B_KING)
        result |= (kg_mask & en_kg).any(dim=1)

        wpa = self.w_pawn_attack_table.T[sq_l]
        bpa = self.b_pawn_attack_table.T[sq_l]
        en_p = torch.where(by_white.unsqueeze(1),
                           wpa & (boards == W_PAWN),
                           bpa & (boards == B_PAWN))
        result |= en_p.any(dim=1)

        occupied = (boards != EMPTY)
        aligned = self.ray_aligned[sq_l]
        rtype = self.ray_type[sq_l]
        between = self.between_mask[sq_l]
        blocked = (between & occupied.unsqueeze(1)).any(dim=2)

        diag_cand = (rtype == 1) & aligned & ~blocked
        en_diag = torch.where(by_white.unsqueeze(1),
                              (boards == W_BISHOP) | (boards == W_QUEEN),
                              (boards == B_BISHOP) | (boards == B_QUEEN))
        result |= (diag_cand & en_diag).any(dim=1)

        str_cand = (rtype == 2) & aligned & ~blocked
        en_str = torch.where(by_white.unsqueeze(1),
                             (boards == W_ROOK) | (boards == W_QUEEN),
                             (boards == B_ROOK) | (boards == B_QUEEN))
        result |= (str_cand & en_str).any(dim=1)

        return result

    def _is_in_check_batched(self, boards, is_white):
        my_king = torch.where(is_white.unsqueeze(1), boards == W_KING, boards == B_KING)
        has_king = my_king.any(dim=1)
        king_sq = my_king.float().argmax(dim=1)
        return self._is_sq_attacked_batched(boards, king_sq, ~is_white) | ~has_king

    # ══════════════════════════════════════════════
    #  BATCHED PSEUDO-LEGAL MOVE GEN
    #  (v9: castling through check fix)
    # ══════════════════════════════════════════════

    def _gen_pseudo_legal_batched(self, boards, is_white, ep, castling):
        M = boards.shape[0]; dev = boards.device
        mask = torch.zeros(M, 64, 64, dtype=torch.bool, device=dev)

        is_wp = (boards >= 1) & (boards <= 6)
        is_bp = (boards >= 7) & (boards <= 12)
        own = torch.where(is_white.unsqueeze(1), is_wp, is_bp)
        enemy = torch.where(is_white.unsqueeze(1), is_bp, is_wp)
        empty = (boards == EMPTY)
        not_own = ~own

        own_n = torch.where(is_white.unsqueeze(1), boards == W_KNIGHT, boards == B_KNIGHT)
        if own_n.any():
            kt = self.knight_attack_table.unsqueeze(0)
            mask |= (own_n.unsqueeze(2) & kt & not_own.unsqueeze(1))

        own_k = torch.where(is_white.unsqueeze(1), boards == W_KING, boards == B_KING)
        if own_k.any():
            kg = self.king_attack_table.unsqueeze(0)
            mask |= (own_k.unsqueeze(2) & kg & not_own.unsqueeze(1))

        own_b = torch.where(is_white.unsqueeze(1), boards == W_BISHOP, boards == B_BISHOP)
        own_r = torch.where(is_white.unsqueeze(1), boards == W_ROOK, boards == B_ROOK)
        own_q = torch.where(is_white.unsqueeze(1), boards == W_QUEEN, boards == B_QUEEN)
        diag_m = own_b | own_q
        str_m = own_r | own_q

        if diag_m.any() or str_m.any():
            occupied = (boards != EMPTY)
            between = self.between_mask.unsqueeze(0)
            occ_exp = occupied.unsqueeze(1).unsqueeze(1)
            blocked = (between & occ_exp).any(dim=-1)
            aligned = self.ray_aligned.unsqueeze(0)
            rtype = self.ray_type.unsqueeze(0)
            can_reach = aligned & ~blocked
            tgt_ok = not_own.unsqueeze(1)
            if diag_m.any():
                mask |= (diag_m.unsqueeze(2) & (rtype == 1) & can_reach & tgt_ok)
            if str_m.any():
                mask |= (str_m.unsqueeze(2) & (rtype == 2) & can_reach & tgt_ok)

        own_p = torch.where(is_white.unsqueeze(1), boards == W_PAWN, boards == B_PAWN)
        if own_p.any():
            iw = is_white
            wf1 = self.tables.w_pawn_fwd1.long(); bf1 = self.tables.b_pawn_fwd1.long()
            wf2 = self.tables.w_pawn_fwd2.long(); bf2 = self.tables.b_pawn_fwd2.long()
            wc = self.tables.w_pawn_caps.long();  bc = self.tables.b_pawn_caps.long()

            f1 = torch.where(iw.unsqueeze(1), wf1.unsqueeze(0).expand(M, -1), bf1.unsqueeze(0).expand(M, -1))
            f1c = f1.clamp(min=0)
            f1_empty = torch.gather(empty.long(), 1, f1c).bool()
            f1_ok = own_p & (f1 >= 0) & f1_empty
            if f1_ok.any():
                idx_arr = f1_ok.nonzero(as_tuple=False)
                mask[idx_arr[:, 0], idx_arr[:, 1], f1[idx_arr[:, 0], idx_arr[:, 1]]] = True

            f2 = torch.where(iw.unsqueeze(1), wf2.unsqueeze(0).expand(M, -1), bf2.unsqueeze(0).expand(M, -1))
            f2c = f2.clamp(min=0)
            f2_empty = torch.gather(empty.long(), 1, f2c).bool()
            f2_ok = f1_ok & (f2 >= 0) & f2_empty
            if f2_ok.any():
                idx_arr = f2_ok.nonzero(as_tuple=False)
                mask[idx_arr[:, 0], idx_arr[:, 1], f2[idx_arr[:, 0], idx_arr[:, 1]]] = True

            for ci in range(2):
                cw = wc[:, ci].unsqueeze(0).expand(M, -1)
                cb = bc[:, ci].unsqueeze(0).expand(M, -1)
                cap = torch.where(iw.unsqueeze(1), cw, cb)
                capc = cap.clamp(min=0)
                has_en = torch.gather(enemy.long(), 1, capc).bool()
                ep_match = (cap == ep.unsqueeze(1).long()) & (ep.unsqueeze(1) >= 0)
                c_ok = own_p & (cap >= 0) & (has_en | ep_match)
                if c_ok.any():
                    idx_arr = c_ok.nonzero(as_tuple=False)
                    mask[idx_arr[:, 0], idx_arr[:, 1], cap[idx_arr[:, 0], idx_arr[:, 1]]] = True

        # v9: Castling with transit-square check verification
        w_env = is_white.nonzero(as_tuple=True)[0]
        if w_env.shape[0] > 0:
            bw = boards[w_env]; cw = castling[w_env]
            on_e1 = bw[:, 4] == W_KING
            # Kingside: e1→g1, transit f1
            ks_cond = on_e1 & cw[:, 0] & (bw[:, 5] == EMPTY) & (bw[:, 6] == EMPTY) & (bw[:, 7] == W_ROOK)
            if ks_cond.any():
                ks_idx = w_env[ks_cond]
                ks_boards = boards[ks_idx]
                by_black = torch.zeros(ks_idx.shape[0], dtype=torch.bool, device=dev)
                # King not in check on e1
                e1_sq = torch.full((ks_idx.shape[0],), 4, dtype=torch.long, device=dev)
                not_in_check = ~self._is_sq_attacked_batched(ks_boards, e1_sq, ~by_black)
                # Transit f1 not attacked
                f1_sq = torch.full((ks_idx.shape[0],), 5, dtype=torch.long, device=dev)
                f1_safe = ~self._is_sq_attacked_batched(ks_boards, f1_sq, ~by_black)
                valid = not_in_check & f1_safe
                if valid.any():
                    mask[ks_idx[valid], 4, 6] = True
            # Queenside: e1→c1, transit d1
            qs_cond = on_e1 & cw[:, 1] & (bw[:, 3] == EMPTY) & (bw[:, 2] == EMPTY) & (bw[:, 1] == EMPTY) & (bw[:, 0] == W_ROOK)
            if qs_cond.any():
                qs_idx = w_env[qs_cond]
                qs_boards = boards[qs_idx]
                by_black = torch.zeros(qs_idx.shape[0], dtype=torch.bool, device=dev)
                e1_sq = torch.full((qs_idx.shape[0],), 4, dtype=torch.long, device=dev)
                not_in_check = ~self._is_sq_attacked_batched(qs_boards, e1_sq, ~by_black)
                d1_sq = torch.full((qs_idx.shape[0],), 3, dtype=torch.long, device=dev)
                d1_safe = ~self._is_sq_attacked_batched(qs_boards, d1_sq, ~by_black)
                valid = not_in_check & d1_safe
                if valid.any():
                    mask[qs_idx[valid], 4, 2] = True

        b_env = (~is_white).nonzero(as_tuple=True)[0]
        if b_env.shape[0] > 0:
            bb = boards[b_env]; cb = castling[b_env]
            on_e8 = bb[:, 60] == B_KING
            # Kingside: e8→g8, transit f8
            ks_cond = on_e8 & cb[:, 2] & (bb[:, 61] == EMPTY) & (bb[:, 62] == EMPTY) & (bb[:, 63] == B_ROOK)
            if ks_cond.any():
                ks_idx = b_env[ks_cond]
                ks_boards = boards[ks_idx]
                by_white = torch.ones(ks_idx.shape[0], dtype=torch.bool, device=dev)
                e8_sq = torch.full((ks_idx.shape[0],), 60, dtype=torch.long, device=dev)
                not_in_check = ~self._is_sq_attacked_batched(ks_boards, e8_sq, by_white)
                f8_sq = torch.full((ks_idx.shape[0],), 61, dtype=torch.long, device=dev)
                f8_safe = ~self._is_sq_attacked_batched(ks_boards, f8_sq, by_white)
                valid = not_in_check & f8_safe
                if valid.any():
                    mask[ks_idx[valid], 60, 62] = True
            # Queenside: e8→c8, transit d8
            qs_cond = on_e8 & cb[:, 3] & (bb[:, 59] == EMPTY) & (bb[:, 58] == EMPTY) & (bb[:, 57] == EMPTY) & (bb[:, 56] == B_ROOK)
            if qs_cond.any():
                qs_idx = b_env[qs_cond]
                qs_boards = boards[qs_idx]
                by_white = torch.ones(qs_idx.shape[0], dtype=torch.bool, device=dev)
                e8_sq = torch.full((qs_idx.shape[0],), 60, dtype=torch.long, device=dev)
                not_in_check = ~self._is_sq_attacked_batched(qs_boards, e8_sq, by_white)
                d8_sq = torch.full((qs_idx.shape[0],), 59, dtype=torch.long, device=dev)
                d8_safe = ~self._is_sq_attacked_batched(qs_boards, d8_sq, by_white)
                valid = not_in_check & d8_safe
                if valid.any():
                    mask[qs_idx[valid], 60, 58] = True

        return mask.view(M, 4096)

    # ══════════════════════════════════════════════
    #  BATCHED KING SAFETY FILTER
    # ══════════════════════════════════════════════

    def _filter_king_safety_batched(self, boards, pseudo_mask, is_white, ep):
        M = boards.shape[0]; dev = boards.device
        move_idx = pseudo_mask.nonzero(as_tuple=False)
        if move_idx.shape[0] == 0: return pseudo_mask
        env_i = move_idx[:, 0]; act_i = move_idx[:, 1]
        from_sq = act_i // 64; to_sq = act_i % 64
        K = env_i.shape[0]; kr = torch.arange(K, device=dev)

        nb = boards[env_i].clone()
        miw = is_white[env_i]; mep = ep[env_i]
        mp = nb[kr, from_sq]
        pt = torch.where(mp <= 6, mp - 1, mp - 7).clamp(min=0).long()

        is_pawn = (pt == 0)
        ep_cap = is_pawn & (to_sq == mep.long()) & (nb[kr, to_sq] == EMPTY)
        if ep_cap.any():
            ei = kr[ep_cap]
            cap_sq = (to_sq[ep_cap] % 8) + (from_sq[ep_cap] // 8) * 8
            nb[ei, cap_sq] = EMPTY

        ik = (pt == 5)
        for (cond_w, fr, to, rsrc, rdst) in [
            (True, 4, 6, 7, 5), (True, 4, 2, 0, 3),
            (False, 60, 62, 63, 61), (False, 60, 58, 56, 59)]:
            sel = ik & (miw if cond_w else ~miw) & (from_sq == fr) & (to_sq == to)
            if sel.any():
                si = kr[sel]
                nb[si, rdst] = nb[si, rsrc]; nb[si, rsrc] = EMPTY

        nb[kr, to_sq] = nb[kr, from_sq]; nb[kr, from_sq] = EMPTY

        promo = is_pawn & ((miw & (to_sq // 8 == 7)) | (~miw & (to_sq // 8 == 0)))
        if promo.any():
            pi = kr[promo]
            nb[pi, to_sq[promo]] = torch.where(miw[promo],
                torch.tensor(W_QUEEN, dtype=torch.int8, device=dev),
                torch.tensor(B_QUEEN, dtype=torch.int8, device=dev))

        mkc = torch.where(miw, torch.tensor(W_KING, dtype=torch.int8, device=dev),
                                torch.tensor(B_KING, dtype=torch.int8, device=dev))
        is_mk = (nb == mkc.unsqueeze(1))
        has_k = is_mk.any(dim=1)
        ksq = is_mk.float().argmax(dim=1)

        attacked = self._is_sq_attacked_batched(nb, ksq, ~miw)
        illegal = attacked | ~has_k

        result = pseudo_mask.clone()
        if illegal.any():
            result[env_i[illegal], act_i[illegal]] = False
        return result

    # ══════════════════════════════════════════════
    #  LEGAL MOVE MASK (v9: vectorized parry)
    # ══════════════════════════════════════════════

    def get_legal_mask(self):
        N, dev = self.N, self.device
        mask = torch.zeros(N, 4099, dtype=torch.bool, device=dev)

        in_def = self.phase == PHASE_DEFENSE
        if in_def.any():
            mask[in_def, ACTION_ATTEMPT_BLOCK] = True
            mask[in_def, ACTION_ATTEMPT_PARRY] = True
            mask[in_def, ACTION_ACCEPT_LOSS] = True

        in_move = self.phase == PHASE_MOVE
        if in_move.any():
            idx = in_move.nonzero(as_tuple=True)[0]
            pseudo = self._gen_pseudo_legal_batched(self.board[idx], self.turn_is_white[idx],
                                                     self.en_passant[idx], self.castling[idx])
            legal = self._filter_king_safety_batched(self.board[idx], pseudo,
                                                      self.turn_is_white[idx], self.en_passant[idx])
            mask[in_move, :4096] = legal

        in_parry = self.phase == PHASE_PARRY
        if in_parry.any():
            idx = in_parry.nonzero(as_tuple=True)[0]
            pf = self._gen_parry_legal_batched(idx)
            mask[in_parry, :4096] = pf

        return mask

    def _gen_parry_legal_batched(self, idx):
        """Generate legal moves for parry phase — fully vectorized (v9)."""
        M = idx.shape[0]
        dev = self.device

        p_sq = self.parry_square[idx].long()
        bds = self.board[idx]
        ctrl_w = self.parry_controller_is_white[idx]
        kr = torch.arange(M, device=dev)
        pcs = bds[kr, p_sq]
        pc_is_w = (pcs >= 1) & (pcs <= 6)
        pt = torch.where(pcs <= 6, pcs - 1, pcs - 7).clamp(min=0).long()

        reachable = torch.zeros(M, 64, dtype=torch.bool, device=dev)

        # Knights
        is_knight = pt == 1
        if is_knight.any():
            ki = is_knight.nonzero(as_tuple=True)[0]
            reachable[ki] = self.knight_attack_table[p_sq[ki]]

        # Kings
        is_king = pt == 5
        if is_king.any():
            ki = is_king.nonzero(as_tuple=True)[0]
            reachable[ki] = self.king_attack_table[p_sq[ki]]

        # Sliding pieces (Bishop=2, Rook=3, Queen=4)
        is_slider = (pt >= 2) & (pt <= 4)
        if is_slider.any():
            si = is_slider.nonzero(as_tuple=True)[0]
            ssq = p_sq[si]
            sbd = bds[si]
            occupied = sbd != EMPTY
            aligned = self.ray_aligned[ssq]
            rtype = self.ray_type[ssq]
            between = self.between_mask[ssq]
            blocked = (between & occupied.unsqueeze(1)).any(dim=2)
            can_reach = aligned & ~blocked

            is_bishop = (pt[si] == 2).unsqueeze(1)
            is_rook = (pt[si] == 3).unsqueeze(1)
            is_queen = (pt[si] == 4).unsqueeze(1)
            diag = (rtype == 1)
            straight = (rtype == 2)

            reach_s = ((is_bishop & diag) | (is_rook & straight) | is_queen) & can_reach
            reachable[si] = reach_s

        # Pawns
        is_pawn = pt == 0
        if is_pawn.any():
            pi = is_pawn.nonzero(as_tuple=True)[0]
            psq = p_sq[pi]
            pbd = bds[pi]
            piw = pc_is_w[pi]
            P = pi.shape[0]

            wf1 = self.tables.w_pawn_fwd1[psq].long()
            bf1 = self.tables.b_pawn_fwd1[psq].long()
            fwd = torch.where(piw, wf1, bf1)
            valid_fwd = fwd >= 0
            if valid_fwd.any():
                fwd_c = fwd.clamp(min=0)
                fwd_empty = pbd[torch.arange(P, device=dev), fwd_c] == EMPTY
                can_fwd = valid_fwd & fwd_empty
                if can_fwd.any():
                    reachable[pi[can_fwd], fwd[can_fwd]] = True

            for ci in range(2):
                wc = self.tables.w_pawn_caps[psq, ci].long()
                bc = self.tables.b_pawn_caps[psq, ci].long()
                cap = torch.where(piw, wc, bc)
                valid_cap = cap >= 0
                if valid_cap.any():
                    reachable[pi[valid_cap], cap[valid_cap]] = True

        # Filter: only empty squares or controller's own pieces (self-capture)
        empty_mask = bds == EMPTY
        ctrl_own = torch.where(ctrl_w.unsqueeze(1),
                               (bds >= 1) & (bds <= 6),
                               (bds >= 7) & (bds <= 12))
        reachable &= (empty_mask | ctrl_own)

        # Convert to flat 4096 mask: action = p_sq * 64 + to_sq
        from_offset = p_sq * 64
        to_all = torch.arange(64, device=dev).unsqueeze(0).expand(M, -1)
        flat_indices = from_offset.unsqueeze(1) + to_all
        pm = torch.zeros(M, 4096, dtype=torch.bool, device=dev)
        pm.scatter_(1, flat_indices, reachable)

        # King safety filter
        pf = self._filter_king_safety_batched(bds, pm, pc_is_w, self.en_passant[idx])

        # Always allow skip (from == to)
        skip_action = p_sq * 64 + p_sq
        pf[kr, skip_action] = True

        return pf

    # ══════════════════════════════════════════════
    #  STEP (v9: fully vectorized dispatch)
    # ══════════════════════════════════════════════

    def step(self, actions):
        N, dev = self.N, self.device
        reward = torch.full((N,), REWARD_STEP_PENALTY, device=dev)

        in_def = self.phase == PHASE_DEFENSE
        if in_def.any(): self._resolve_defense_batched(actions, in_def, reward)
        in_move = self.phase == PHASE_MOVE
        if in_move.any(): self._apply_moves_batched(actions, in_move, reward)
        in_parry = self.phase == PHASE_PARRY
        if in_parry.any(): self._apply_parry_batched(actions, in_parry, reward)

        self._check_endgame(reward)

        active = self.phase != PHASE_FINISHED
        self.full_move_count[active] += 1

        over = (self.full_move_count >= self.max_steps * 2) & active
        if over.any():
            self.phase[over] = PHASE_FINISHED; self.result[over] = RESULT_DRAW

        fifty_move = (self.half_moves >= 100) & (self.phase != PHASE_FINISHED)
        if fifty_move.any():
            self.phase[fifty_move] = PHASE_FINISHED
            self.result[fifty_move] = RESULT_DRAW

        # v9: threefold repetition detection
        still_playing = (self.phase == PHASE_MOVE)
        if still_playing.any():
            sp_idx = still_playing.nonzero(as_tuple=True)[0]
            self._update_zobrist(sp_idx)
            threefold = self._check_threefold(sp_idx)
            if threefold.any():
                tf_idx = sp_idx[threefold]
                self.phase[tf_idx] = PHASE_FINISHED
                self.result[tf_idx] = RESULT_DRAW

        done = self.phase == PHASE_FINISHED

        info = {
            "result": self.result.clone(),
            "full_move_count": self.full_move_count.clone(),
            "check_attempts": self.check_attempts.clone(),
        }

        if done.any():
            reward[done] += reward_terminal(
                self.result,
                self.agent_is_white,
                board=self.board,
                piece_values=self.tables.piece_values,
                full_move_count=self.full_move_count,
                max_steps=self.max_steps
            )[done]
            self.reset(done)

        return self._build_obs(), reward, done, info

    # ══════════════════════════════════════════════
    #  VECTORIZED DEFENSE RESOLUTION (v9)
    # ══════════════════════════════════════════════

    def _resolve_defense_batched(self, actions, mask, reward):
        indices = mask.nonzero(as_tuple=True)[0]
        M = indices.shape[0]
        if M == 0: return
        dev = self.device

        # Any defense resolution consumes the en-passant opportunity from the
        # previous move, regardless of block/parry/capture outcome.
        self.en_passant[indices] = -1

        a = actions[indices]
        ap = self.pending_attacker_piece[indices]
        dp = self.pending_defender_piece[indices]
        at = torch.where(ap <= 6, ap - 1, ap - 7).clamp(min=0).long()
        dt = torch.where(dp <= 6, dp - 1, dp - 7).clamp(min=0).long()

        atk_stat = self.tables.attack_stats[at]
        def_stat = self.tables.defense_stats[dt]
        tau = def_stat / (atk_stat + def_stat)

        rolls = torch.rand(M, device=dev)

        is_accept = (a == ACTION_ACCEPT_LOSS)
        is_block = (a == ACTION_ATTEMPT_BLOCK)
        is_parry = (a == ACTION_ATTEMPT_PARRY)
        block_success = is_block & (rolls < tau)
        parry_success = is_parry & (rolls < tau)

        outcome_E = is_accept | (is_block & ~block_success) | (is_parry & ~parry_success)
        outcome_B = block_success
        outcome_P = parry_success

        fs = self.pending_attacker_sq[indices].long()
        ts = self.pending_target_sq[indices].long()
        aw = self.pending_attacker_color_white[indices]
        agent_is_attacker = (aw == self.agent_is_white[indices])

        # ── Execute captures (outcome E) ──
        if outcome_E.any():
            ei = indices[outcome_E]
            efs = fs[outcome_E]; ets = ts[outcome_E]
            eaw = aw[outcome_E]

            self.board[ei, ets] = self.board[ei, efs]
            self.board[ei, efs] = EMPTY
            self._post_move_updates_batched(ei, efs, ets)
            self.turn_is_white[ei] = ~eaw
            self.phase[ei] = PHASE_MOVE

            edt = dt[outcome_E]
            dv = self.tables.piece_values[edt]
            eat = at[outcome_E]
            e_aia = agent_is_attacker[outcome_E]
            e_atk_stat = atk_stat[outcome_E]
            e_def_stat = def_stat[outcome_E]

            reward[ei] += torch.where(e_aia,
                                      REWARD_CAPTURE_SCALE * dv,
                                      REWARD_LOSE_PIECE_SCALE * dv)

            bonus_mask = e_aia & (e_atk_stat > e_def_stat)
            if bonus_mask.any():
                bi = ei[bonus_mask]
                reward[bi] += REWARD_CAPTURE_ATTACKER_BONUS * (e_atk_stat[bonus_mask] - e_def_stat[bonus_mask])

            e_accept = is_accept[outcome_E]
            accept_non_atk = e_accept & ~e_aia
            if accept_non_atk.any():
                reward[ei[accept_non_atk]] += REWARD_ACCEPT_LOSS
            fail_non_atk = ~e_accept & ~e_aia
            if fail_non_atk.any():
                reward[ei[fail_non_atk]] += REWARD_DEFENSE_FAIL

            self.half_moves[ei] = 0

            # Stats
            self.capture_total_count += e_aia.sum().item()
            self.capture_high_attacker_count += bonus_mask.sum().item()

        # ── Block success (outcome B) ──
        if outcome_B.any():
            bi = indices[outcome_B]
            baw = aw[outcome_B]
            self.turn_is_white[bi] = ~baw
            self.phase[bi] = PHASE_MOVE
            b_aia = agent_is_attacker[outcome_B]
            reward[bi] += torch.where(~b_aia,
                                      torch.tensor(REWARD_BLOCK_SUCCESS, device=dev),
                                      torch.tensor(-REWARD_BLOCK_SUCCESS * 0.5, device=dev))

        # ── Parry success (outcome P) ──
        if outcome_P.any():
            pi = indices[outcome_P]
            paw = aw[outcome_P]
            pfs = fs[outcome_P]
            self.turn_is_white[pi] = ~paw
            self.phase[pi] = PHASE_PARRY
            self.parry_square[pi] = pfs.short()
            self.parry_controller_is_white[pi] = ~paw
            p_aia = agent_is_attacker[outcome_P]
            reward[pi] += torch.where(~p_aia,
                                      torch.tensor(REWARD_PARRY_SUCCESS, device=dev),
                                      torch.tensor(-REWARD_PARRY_SUCCESS * 0.5, device=dev))

        # Clear pending
        self.pending_attacker_sq[indices] = -1
        self.pending_target_sq[indices] = -1

        # Enforce check for all defense envs
        self._enforce_check_batched(indices, aw, reward)

    # ══════════════════════════════════════════════
    #  VECTORIZED NORMAL MOVES (v9)
    # ══════════════════════════════════════════════

    def _apply_moves_batched(self, actions, mask, reward):
        indices = mask.nonzero(as_tuple=True)[0]
        M = indices.shape[0]
        if M == 0: return
        dev = self.device
        kr = torch.arange(M, device=dev)

        a = actions[indices]
        board_action = a < 4096
        if not board_action.all():
            valid = board_action
            indices = indices[valid]; a = a[valid]
            M = indices.shape[0]
            if M == 0: return
            kr = torch.arange(M, device=dev)

        fs = a // 64; ts = a % 64
        mv = self.board[indices, fs]
        tgt = self.board[indices, ts]

        valid = mv != EMPTY
        if not valid.all():
            indices = indices[valid]; a = a[valid]; fs = fs[valid]; ts = ts[valid]
            mv = mv[valid]; tgt = tgt[valid]
            M = indices.shape[0]
            if M == 0: return
            kr = torch.arange(M, device=dev)

        mw = (mv >= 1) & (mv <= 6)
        pt = torch.where(mv <= 6, mv - 1, mv - 7).clamp(min=0).long()
        is_capture = tgt != EMPTY

        # En passant detection
        is_pawn = pt == 0
        ep = self.en_passant[indices].long()
        is_ep = is_pawn & (ts == ep) & (tgt == EMPTY) & (ep >= 0)

        if is_ep.any():
            ep_cs = (ts[is_ep] % 8) + (fs[is_ep] // 8) * 8
            # Store captured pawn piece before removing (for defense)
            ep_captured = self.board[indices[is_ep], ep_cs].clone()
            self.board[indices[is_ep], ep_cs] = EMPTY
            # Mark as capture, store defender piece
            is_capture = is_capture | is_ep
            # Replace tgt for ep captures with the captured pawn
            tgt = tgt.clone()
            tgt[is_ep] = ep_captured

        # Check escape stats
        actor_ci = torch.where(mw, torch.tensor(0, device=dev), torch.tensor(1, device=dev)).long()
        current_ca = self.check_attempts[indices, actor_ci]
        is_in_check = current_ca > 0

        if is_in_check.any():
            chk_cap = is_in_check & is_capture
            chk_mov = is_in_check & ~is_capture
            self.check_escape_by_capture_count += chk_cap.sum().item()
            self.check_escape_by_move_count += chk_mov.sum().item()
            chk_3rd = is_in_check & (current_ca >= 2)
            self.check_3rd_attempt_capture_count += (chk_3rd & is_capture).sum().item()
            self.check_3rd_attempt_move_count += (chk_3rd & ~is_capture).sum().item()

        # ── Start defense for captures ──
        cap_mask = is_capture
        if cap_mask.any():
            ci = indices[cap_mask]
            self.phase[ci] = PHASE_DEFENSE
            self.pending_attacker_sq[ci] = fs[cap_mask].short()
            self.pending_target_sq[ci] = ts[cap_mask].short()
            self.pending_attacker_piece[ci] = mv[cap_mask]
            self.pending_defender_piece[ci] = tgt[cap_mask]
            self.pending_attacker_color_white[ci] = mw[cap_mask]
            self.turn_is_white[ci] = ~mw[cap_mask]

        # ── Non-captures: apply move ──
        nc_mask = ~is_capture
        if nc_mask.any():
            ni = indices[nc_mask]
            nfs = fs[nc_mask]; nts = ts[nc_mask]
            npt = pt[nc_mask]; nmw = mw[nc_mask]
            nM = ni.shape[0]

            # Reset en passant
            self.en_passant[ni] = -1

            # Castling rook movement
            is_king = npt == 5
            if is_king.any():
                for (cond_w, f, t, rs, rd) in [(True, 4, 6, 7, 5), (True, 4, 2, 0, 3),
                                                (False, 60, 62, 63, 61), (False, 60, 58, 56, 59)]:
                    sel = is_king & (nmw if cond_w else ~nmw) & (nfs == f) & (nts == t)
                    if sel.any():
                        si = ni[sel]
                        self.board[si, rd] = self.board[si, rs]
                        self.board[si, rs] = EMPTY

            # Move piece
            self.board[ni, nts] = self.board[ni, nfs]
            self.board[ni, nfs] = EMPTY

            # En passant creation (double pawn push)
            is_pawn_n = npt == 0
            double_push = is_pawn_n & ((nfs // 8 - nts // 8).abs() == 2)
            if double_push.any():
                di = ni[double_push]
                dfs = nfs[double_push]; dts = nts[double_push]
                ep_sq = (dfs % 8) + ((dfs // 8 + dts // 8) // 2) * 8
                self.en_passant[di] = ep_sq.short()

            # Half moves
            hm = self.half_moves[ni]
            self.half_moves[ni] = torch.where(is_pawn_n,
                                              torch.zeros_like(hm), hm + 1)

            # Post move updates (promotion + castling rights)
            self._post_move_updates_batched(ni, nfs, nts)

            # Switch turn
            self.turn_is_white[ni] = ~nmw

            # Enforce check
            self._enforce_check_batched(ni, nmw, reward)

    # ══════════════════════════════════════════════
    #  VECTORIZED PARRY MOVES (v9)
    # ══════════════════════════════════════════════

    def _apply_parry_batched(self, actions, mask, reward):
        indices = mask.nonzero(as_tuple=True)[0]
        M = indices.shape[0]
        if M == 0: return
        dev = self.device
        kr = torch.arange(M, device=dev)

        a = actions[indices]
        board_action = a < 4096
        if not board_action.all():
            valid = board_action
            indices = indices[valid]; a = a[valid]
            M = indices.shape[0]
            if M == 0: return
            kr = torch.arange(M, device=dev)

        fs = a // 64; ts = a % 64
        cw = self.parry_controller_is_white[indices]
        tgt = self.board[indices, ts]

        is_skip = (fs == ts)
        is_capture = (tgt != EMPTY) & ~is_skip
        is_good = ~is_skip & ~is_capture

        # Stats (batched)
        n_skip = is_skip.sum().item()
        n_capture = is_capture.sum().item()
        n_good = is_good.sum().item()
        n_total = n_skip + n_capture + n_good
        self.parry_skip_count += n_skip
        self.parry_self_capture_count += n_capture
        self.parry_good_move_count += n_good
        self.parry_total_count += n_total

        # ── Skip ──
        if is_skip.any():
            si = indices[is_skip]
            scw = cw[is_skip]
            self.phase[si] = PHASE_MOVE
            self.parry_square[si] = -1
            reward[si] += REWARD_PARRY_SKIP
            self._enforce_check_batched(si, scw, reward)

        # ── Self-capture ──
        if is_capture.any():
            ci = indices[is_capture]
            ccw = cw[is_capture]
            cfs = fs[is_capture]; cts = ts[is_capture]
            ctgt = tgt[is_capture]
            dt_val = torch.where(ctgt <= 6, ctgt - 1, ctgt - 7).clamp(min=0).long()
            piece_val = self.tables.piece_values[dt_val]

            self.board[ci, cts] = self.board[ci, cfs]
            self.board[ci, cfs] = EMPTY
            self._post_move_updates_batched(ci, cfs, cts)
            self.phase[ci] = PHASE_MOVE
            self.parry_square[ci] = -1
            reward[ci] += REWARD_PARRY_SELF_CAPTURE * piece_val
            self.half_moves[ci] = 0
            self._enforce_check_batched(ci, ccw, reward)

        # ── Good move (empty square) ──
        if is_good.any():
            gi = indices[is_good]
            gcw = cw[is_good]
            gfs = fs[is_good]; gts = ts[is_good]

            self.board[gi, gts] = self.board[gi, gfs]
            self.board[gi, gfs] = EMPTY
            self._post_move_updates_batched(gi, gfs, gts)
            self.phase[gi] = PHASE_MOVE
            self.parry_square[gi] = -1
            reward[gi] += REWARD_PARRY_MOVE_GOOD
            self._enforce_check_batched(gi, gcw, reward)

    # ══════════════════════════════════════════════
    #  VECTORIZED POST-MOVE UPDATES (v9)
    # ══════════════════════════════════════════════

    def _post_move_updates_batched(self, indices, from_sqs, to_sqs):
        """Update castling rights and handle pawn promotion — batched."""
        M = indices.shape[0]
        if M == 0: return
        dev = self.device

        pieces = self.board[indices, to_sqs]
        pt = torch.where(pieces <= 6, pieces - 1, pieces - 7).clamp(min=0).long()
        is_white = (pieces >= 1) & (pieces <= 6)

        # King moves: revoke castling
        is_king = (pt == 5)
        kw = is_king & is_white
        kb = is_king & ~is_white
        if kw.any():
            ki = indices[kw]
            self.castling[ki, 0] = False; self.castling[ki, 1] = False
        if kb.any():
            ki = indices[kb]
            self.castling[ki, 2] = False; self.castling[ki, 3] = False

        # Rook moves from corner / captures on corner
        is_rook = (pt == 3)
        for sq, ci in [(0, 1), (7, 0), (56, 3), (63, 2)]:
            rook_from = is_rook & (from_sqs == sq)
            if rook_from.any():
                self.castling[indices[rook_from], ci] = False
            any_to = (to_sqs == sq)
            if any_to.any():
                self.castling[indices[any_to], ci] = False

        # Pawn promotion (always to queen)
        is_pawn = (pt == 0)
        promo_w = is_pawn & is_white & (to_sqs // 8 == 7)
        promo_b = is_pawn & ~is_white & (to_sqs // 8 == 0)
        if promo_w.any():
            pi = indices[promo_w]
            self.board[pi, to_sqs[promo_w]] = W_QUEEN
        if promo_b.any():
            pi = indices[promo_b]
            self.board[pi, to_sqs[promo_b]] = B_QUEEN

    # ══════════════════════════════════════════════
    #  VECTORIZED CHECK ENFORCEMENT (v9)
    # ══════════════════════════════════════════════

    def _enforce_check_batched(self, indices, actor_w, reward):
        """Check enforcement for a batch of envs — no per-env loops."""
        M = indices.shape[0]
        if M == 0: return
        dev = self.device

        boards = self.board[indices]
        actor_in_check = self._is_in_check_batched(boards, actor_w)

        actor_ci = torch.where(actor_w, torch.tensor(0, device=dev),
                               torch.tensor(1, device=dev)).long()

        # ── Envs where actor IS in check ──
        check_mask = actor_in_check
        if check_mask.any():
            ci = indices[check_mask]
            aci = actor_ci[check_mask]

            self.check_attempts[ci, aci] += 1
            new_attempts = self.check_attempts[ci, aci]

            agent_is_actor = (actor_w[check_mask] == self.agent_is_white[ci])
            agent_is_checker = ~agent_is_actor

            # Penalty for being in check / reward for giving check
            reward[ci] += torch.where(agent_is_actor,
                                      torch.tensor(REWARD_CHECK_ATTEMPT_PENALTY, device=dev),
                                      torch.tensor(REWARD_CHECK_GIVEN, device=dev))

            second_plus = new_attempts >= 2
            bonus = agent_is_checker & second_plus
            if bonus.any():
                reward[ci[bonus]] += REWARD_CHECK_2ND_ATTEMPT

            # Turn back to actor to escape check
            self.turn_is_white[ci] = actor_w[check_mask]
            self.phase[ci] = PHASE_MOVE
            self.pending_attacker_sq[ci] = -1
            self.pending_target_sq[ci] = -1
            self.parry_square[ci] = -1

            # 3 check attempts = game over
            three_checks = new_attempts >= 3
            if three_checks.any():
                gi = ci[three_checks]
                gaw = actor_w[check_mask][three_checks]
                self.phase[gi] = PHASE_FINISHED
                self.result[gi] = torch.where(gaw,
                    torch.tensor(RESULT_BLACK_WIN, dtype=torch.int8, device=dev),
                    torch.tensor(RESULT_WHITE_WIN, dtype=torch.int8, device=dev))

        # ── Envs where actor is NOT in check ──
        no_check = ~actor_in_check
        if no_check.any():
            ni = indices[no_check]
            nci = actor_ci[no_check]
            old_attempts = self.check_attempts[ni, nci].clone()
            self.check_attempts[ni, nci] = 0

            # Escape reward
            was_in_check = old_attempts > 0
            agent_is_actor_nc = (actor_w[no_check] == self.agent_is_white[ni])
            give_escape = was_in_check & agent_is_actor_nc
            if give_escape.any():
                ei = ni[give_escape]
                urgency = old_attempts[give_escape].float().clamp(max=2)
                reward[ei] += REWARD_CHECK_ESCAPE_SUCCESS * urgency

    # ══════════════════════════════════════════════
    #  ENDGAME CHECK
    # ══════════════════════════════════════════════

    def _check_endgame(self, reward):
        active = (self.phase == PHASE_MOVE)
        if not active.any(): return
        bds = self.board[active]; ai = active.nonzero(as_tuple=True)[0]
        wm = ~(bds == W_KING).any(dim=1); bm = ~(bds == B_KING).any(dim=1)
        if wm.any(): self.phase[ai[wm]] = PHASE_FINISHED; self.result[ai[wm]] = RESULT_BLACK_WIN
        if bm.any(): self.phase[ai[bm]] = PHASE_FINISHED; self.result[ai[bm]] = RESULT_WHITE_WIN
        sa = (self.phase == PHASE_MOVE)
        if not sa.any(): return
        si = sa.nonzero(as_tuple=True)[0]
        pseudo = self._gen_pseudo_legal_batched(self.board[si], self.turn_is_white[si], self.en_passant[si], self.castling[si])
        legal = self._filter_king_safety_batched(self.board[si], pseudo, self.turn_is_white[si], self.en_passant[si])
        nm = ~legal.any(dim=1)
        if nm.any():
            ni = si[nm]; ic = self._is_in_check_batched(self.board[ni], self.turn_is_white[ni])
            self.phase[ni] = PHASE_FINISHED; iw = self.turn_is_white[ni]
            self.result[ni] = torch.where(ic,
                torch.where(iw, torch.tensor(RESULT_BLACK_WIN, dtype=torch.int8, device=self.device),
                                 torch.tensor(RESULT_WHITE_WIN, dtype=torch.int8, device=self.device)),
                torch.tensor(RESULT_DRAW, dtype=torch.int8, device=self.device))
