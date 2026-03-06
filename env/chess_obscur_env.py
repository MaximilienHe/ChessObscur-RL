"""
chess_obscur_env.py — Fully vectorized GPU Chess Obscur environment.

CHANGES v5 (parry rules fix):
  FIX: Removed illegal "enemy capture" during parry phase.
  During parry, the controller moves the attacker's piece. Legal outcomes are:
    1. SKIP (from == to): don't move the piece
    2. GOOD MOVE: move to an empty square
    3. SELF-CAPTURE: move onto controller's own piece (bad)
  Moving the parried piece onto another piece of the same color as the parried
  piece is NOT allowed (it would mean eating your own teammate, which makes no
  sense from the attacker's perspective and is not in the game rules).
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
    REWARD_CHECK_ESCAPE_SUCCESS,  # v6: outcome-based, replaces action-type rewards
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


class ChessObscurEnv:
    def __init__(self, num_envs: int, device: str = "cuda", max_steps: int = 150):
        self.N = num_envs
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.tables = MoveTables(device)
        self._precompute_attack_tables()

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

        # ── Parry outcome counters (3 outcomes only: skip, good_move, self_capture) ──
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
        self.agent_is_white[mask] = torch.rand(n, device=dev) > 0.5
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
    #  OBSERVATION
    # ══════════════════════════════════════════════

    def _build_obs(self):
        N, dev = self.N, self.device
        obs = torch.zeros(N, 19, 8, 8, device=dev)
        board = self.board
        is_w = self.turn_is_white.view(N,1,1).float()
        for pt in range(6):
            wm = (board==pt+1).float().view(N,8,8)
            bm = (board==pt+7).float().view(N,8,8)
            obs[:,pt] = wm*is_w + bm*(1-is_w)
            obs[:,6+pt] = bm*is_w + wm*(1-is_w)
        ep = self.en_passant; ev = ep >= 0
        if ev.any():
            ef = torch.zeros(N,64,device=dev)
            ef.scatter_(1, ep.long().clamp(0,63).unsqueeze(1), ev.float().unsqueeze(1))
            obs[:,12] = ef.view(N,8,8)
        c = self.castling.float(); iw = self.turn_is_white.float()
        obs[:,13,0,:] = (c[:,0]*iw+c[:,2]*(1-iw)).unsqueeze(1).expand(-1,8)
        obs[:,13,1,:] = (c[:,1]*iw+c[:,3]*(1-iw)).unsqueeze(1).expand(-1,8)
        obs[:,13,2,:] = (c[:,2]*iw+c[:,0]*(1-iw)).unsqueeze(1).expand(-1,8)
        obs[:,13,3,:] = (c[:,3]*iw+c[:,1]*(1-iw)).unsqueeze(1).expand(-1,8)
        obs[:,14] = self.turn_is_white.float().view(N,1,1).expand(-1,8,8)
        obs[:,15] = self._is_in_check_batched(self.board, self.turn_is_white).float().view(N,1,1).expand(-1,8,8)
        obs[:,16] = (self.phase.float()/3).view(N,1,1).expand(-1,8,8)
        ca = self.check_attempts
        obs[:,17] = torch.where(self.turn_is_white, ca[:,0], ca[:,1]).float().div(3).view(N,1,1).expand(-1,8,8)
        obs[:,18] = (self.half_moves.float()/100).view(N,1,1).expand(-1,8,8)
        return obs

    # ══════════════════════════════════════════════
    #  BATCHED ATTACK DETECTION
    # ══════════════════════════════════════════════

    def _is_sq_attacked_batched(self, boards, sq, by_white):
        M = boards.shape[0]; dev = boards.device
        result = torch.zeros(M, dtype=torch.bool, device=dev)
        sq_l = sq.long()

        kt_mask = self.knight_attack_table[sq_l]
        en_k = torch.where(by_white.unsqueeze(1), boards==W_KNIGHT, boards==B_KNIGHT)
        result |= (kt_mask & en_k).any(dim=1)

        kg_mask = self.king_attack_table[sq_l]
        en_kg = torch.where(by_white.unsqueeze(1), boards==W_KING, boards==B_KING)
        result |= (kg_mask & en_kg).any(dim=1)

        wpa = self.w_pawn_attack_table.T[sq_l]
        bpa = self.b_pawn_attack_table.T[sq_l]
        en_p = torch.where(by_white.unsqueeze(1),
                           wpa & (boards==W_PAWN),
                           bpa & (boards==B_PAWN))
        result |= en_p.any(dim=1)

        occupied = (boards != EMPTY)
        aligned = self.ray_aligned[sq_l]
        rtype = self.ray_type[sq_l]
        between = self.between_mask[sq_l]
        blocked = (between & occupied.unsqueeze(1)).any(dim=2)

        diag_cand = (rtype==1) & aligned & ~blocked
        en_diag = torch.where(by_white.unsqueeze(1),
                              (boards==W_BISHOP)|(boards==W_QUEEN),
                              (boards==B_BISHOP)|(boards==B_QUEEN))
        result |= (diag_cand & en_diag).any(dim=1)

        str_cand = (rtype==2) & aligned & ~blocked
        en_str = torch.where(by_white.unsqueeze(1),
                             (boards==W_ROOK)|(boards==W_QUEEN),
                             (boards==B_ROOK)|(boards==B_QUEEN))
        result |= (str_cand & en_str).any(dim=1)

        return result

    def _is_in_check_batched(self, boards, is_white):
        my_king = torch.where(is_white.unsqueeze(1), boards==W_KING, boards==B_KING)
        has_king = my_king.any(dim=1)
        king_sq = my_king.float().argmax(dim=1)
        return self._is_sq_attacked_batched(boards, king_sq, ~is_white) | ~has_king

    # ══════════════════════════════════════════════
    #  BATCHED PSEUDO-LEGAL MOVE GEN
    # ══════════════════════════════════════════════

    def _gen_pseudo_legal_batched(self, boards, is_white, ep, castling):
        M = boards.shape[0]; dev = boards.device
        mask = torch.zeros(M, 64, 64, dtype=torch.bool, device=dev)

        is_wp = (boards>=1)&(boards<=6)
        is_bp = (boards>=7)&(boards<=12)
        own = torch.where(is_white.unsqueeze(1), is_wp, is_bp)
        enemy = torch.where(is_white.unsqueeze(1), is_bp, is_wp)
        empty = (boards==EMPTY)
        not_own = ~own

        own_n = torch.where(is_white.unsqueeze(1), boards==W_KNIGHT, boards==B_KNIGHT)
        if own_n.any():
            kt = self.knight_attack_table.unsqueeze(0)
            mask |= (own_n.unsqueeze(2) & kt & not_own.unsqueeze(1))

        own_k = torch.where(is_white.unsqueeze(1), boards==W_KING, boards==B_KING)
        if own_k.any():
            kg = self.king_attack_table.unsqueeze(0)
            mask |= (own_k.unsqueeze(2) & kg & not_own.unsqueeze(1))

        own_b = torch.where(is_white.unsqueeze(1), boards==W_BISHOP, boards==B_BISHOP)
        own_r = torch.where(is_white.unsqueeze(1), boards==W_ROOK, boards==B_ROOK)
        own_q = torch.where(is_white.unsqueeze(1), boards==W_QUEEN, boards==B_QUEEN)
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
                mask |= (diag_m.unsqueeze(2) & (rtype==1) & can_reach & tgt_ok)
            if str_m.any():
                mask |= (str_m.unsqueeze(2) & (rtype==2) & can_reach & tgt_ok)

        own_p = torch.where(is_white.unsqueeze(1), boards==W_PAWN, boards==B_PAWN)
        if own_p.any():
            iw = is_white
            wf1=self.tables.w_pawn_fwd1.long(); bf1=self.tables.b_pawn_fwd1.long()
            wf2=self.tables.w_pawn_fwd2.long(); bf2=self.tables.b_pawn_fwd2.long()
            wc=self.tables.w_pawn_caps.long();  bc=self.tables.b_pawn_caps.long()

            f1 = torch.where(iw.unsqueeze(1), wf1.unsqueeze(0).expand(M,-1), bf1.unsqueeze(0).expand(M,-1))
            f1c = f1.clamp(min=0)
            f1_empty = torch.gather(empty.long(),1,f1c).bool()
            f1_ok = own_p & (f1>=0) & f1_empty
            if f1_ok.any():
                idx_arr = f1_ok.nonzero(as_tuple=False)
                mask[idx_arr[:,0], idx_arr[:,1], f1[idx_arr[:,0],idx_arr[:,1]]] = True

            f2 = torch.where(iw.unsqueeze(1), wf2.unsqueeze(0).expand(M,-1), bf2.unsqueeze(0).expand(M,-1))
            f2c = f2.clamp(min=0)
            f2_empty = torch.gather(empty.long(),1,f2c).bool()
            f2_ok = f1_ok & (f2>=0) & f2_empty
            if f2_ok.any():
                idx_arr = f2_ok.nonzero(as_tuple=False)
                mask[idx_arr[:,0], idx_arr[:,1], f2[idx_arr[:,0],idx_arr[:,1]]] = True

            for ci in range(2):
                cw = wc[:,ci].unsqueeze(0).expand(M,-1)
                cb = bc[:,ci].unsqueeze(0).expand(M,-1)
                cap = torch.where(iw.unsqueeze(1), cw, cb)
                capc = cap.clamp(min=0)
                has_en = torch.gather(enemy.long(),1,capc).bool()
                ep_match = (cap == ep.unsqueeze(1).long()) & (ep.unsqueeze(1)>=0)
                c_ok = own_p & (cap>=0) & (has_en | ep_match)
                if c_ok.any():
                    idx_arr = c_ok.nonzero(as_tuple=False)
                    mask[idx_arr[:,0], idx_arr[:,1], cap[idx_arr[:,0],idx_arr[:,1]]] = True

        w_env = is_white.nonzero(as_tuple=True)[0]
        if w_env.shape[0] > 0:
            bw = boards[w_env]; cw = castling[w_env]
            on_e1 = bw[:,4]==W_KING
            ks = on_e1 & cw[:,0] & (bw[:,5]==EMPTY) & (bw[:,6]==EMPTY) & (bw[:,7]==W_ROOK)
            qs = on_e1 & cw[:,1] & (bw[:,3]==EMPTY) & (bw[:,2]==EMPTY) & (bw[:,1]==EMPTY) & (bw[:,0]==W_ROOK)
            if ks.any(): mask[w_env[ks], 4, 6] = True
            if qs.any(): mask[w_env[qs], 4, 2] = True

        b_env = (~is_white).nonzero(as_tuple=True)[0]
        if b_env.shape[0] > 0:
            bb = boards[b_env]; cb = castling[b_env]
            on_e8 = bb[:,60]==B_KING
            ks = on_e8 & cb[:,2] & (bb[:,61]==EMPTY) & (bb[:,62]==EMPTY) & (bb[:,63]==B_ROOK)
            qs = on_e8 & cb[:,3] & (bb[:,59]==EMPTY) & (bb[:,58]==EMPTY) & (bb[:,57]==EMPTY) & (bb[:,56]==B_ROOK)
            if ks.any(): mask[b_env[ks], 60, 62] = True
            if qs.any(): mask[b_env[qs], 60, 58] = True

        return mask.view(M, 4096)

    # ══════════════════════════════════════════════
    #  BATCHED KING SAFETY FILTER
    # ══════════════════════════════════════════════

    def _filter_king_safety_batched(self, boards, pseudo_mask, is_white, ep):
        M = boards.shape[0]; dev = boards.device
        move_idx = pseudo_mask.nonzero(as_tuple=False)
        if move_idx.shape[0] == 0: return pseudo_mask
        env_i = move_idx[:,0]; act_i = move_idx[:,1]
        from_sq = act_i // 64; to_sq = act_i % 64
        K = env_i.shape[0]; kr = torch.arange(K, device=dev)

        nb = boards[env_i].clone()
        miw = is_white[env_i]; mep = ep[env_i]
        mp = nb[kr, from_sq]
        pt = torch.where(mp<=6, mp-1, mp-7).clamp(min=0).long()

        is_pawn = (pt==0)
        ep_cap = is_pawn & (to_sq==mep.long()) & (nb[kr, to_sq]==EMPTY)
        if ep_cap.any():
            ei = kr[ep_cap]
            cap_sq = (to_sq[ep_cap]%8) + (from_sq[ep_cap]//8)*8
            nb[ei, cap_sq] = EMPTY

        ik = (pt==5)
        for (cond_w, fr, to, rsrc, rdst) in [
            (True, 4, 6, 7, 5), (True, 4, 2, 0, 3),
            (False, 60, 62, 63, 61), (False, 60, 58, 56, 59)]:
            sel = ik & (miw if cond_w else ~miw) & (from_sq==fr) & (to_sq==to)
            if sel.any():
                si = kr[sel]
                nb[si, rdst] = nb[si, rsrc]; nb[si, rsrc] = EMPTY

        nb[kr, to_sq] = nb[kr, from_sq]; nb[kr, from_sq] = EMPTY

        promo = is_pawn & ((miw & (to_sq//8==7)) | (~miw & (to_sq//8==0)))
        if promo.any():
            pi = kr[promo]
            nb[pi, to_sq[promo]] = torch.where(miw[promo],
                torch.tensor(W_QUEEN, dtype=torch.int8, device=dev),
                torch.tensor(B_QUEEN, dtype=torch.int8, device=dev))

        mkc = torch.where(miw, torch.tensor(W_KING,dtype=torch.int8,device=dev),
                                torch.tensor(B_KING,dtype=torch.int8,device=dev))
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
    #  LEGAL MOVE MASK
    # ══════════════════════════════════════════════

    def get_legal_mask(self):
        N, dev = self.N, self.device
        mask = torch.zeros(N, 4163, dtype=torch.bool, device=dev)

        in_def = self.phase == PHASE_DEFENSE
        if in_def.any():
            mask[in_def, 4160] = True; mask[in_def, 4161] = True; mask[in_def, 4162] = True

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
            M = idx.shape[0]
            p_sq = self.parry_square[idx].long()
            bds = self.board[idx]
            ctrl_w = self.parry_controller_is_white[idx]
            pcs = bds[torch.arange(M, device=dev), p_sq]
            pc_is_w = (pcs>=1)&(pcs<=6)

            # ── v5 FIX: Build parry legal mask manually ──
            # During parry, the controller moves the parried piece.
            # Legal targets: empty squares OR controller's own pieces (self-capture).
            # NOT legal: pieces of the same color as the parried piece (enemy capture is illegal).

            pm = torch.zeros(M, 64, 64, dtype=torch.bool, device=dev)

            for mi in range(M):
                sq = p_sq[mi].item()
                pc = pcs[mi].item()
                if pc == EMPTY:
                    continue

                pty = (pc - 1) if pc <= 6 else (pc - 7)
                piece_is_white = (1 <= pc <= 6)
                controller_is_white = ctrl_w[mi].item()

                # Get reachable squares for this piece type
                if pty == 1:  # Knight
                    tgts = self.knight_attack_table[sq]
                elif pty == 5:  # King
                    tgts = self.king_attack_table[sq]
                elif pty in (2, 3, 4):  # Bishop, Rook, Queen
                    occ = bds[mi] != EMPTY
                    al = self.ray_aligned[sq]
                    rt = self.ray_type[sq]
                    blk = (self.between_mask[sq] & occ.unsqueeze(0)).any(dim=1)
                    if pty == 2:    # Bishop
                        tgts = al & (rt == 1) & ~blk
                    elif pty == 3:  # Rook
                        tgts = al & (rt == 2) & ~blk
                    else:           # Queen
                        tgts = al & ~blk
                elif pty == 0:  # Pawn
                    # Pawn in parry: forward moves + capture diagonals
                    # (captures here = self-capture onto controller's pieces)
                    tgts = torch.zeros(64, dtype=torch.bool, device=dev)
                    if piece_is_white:
                        fw = self.tables.w_pawn_fwd1[sq].item()
                        ct = self.tables.w_pawn_caps
                    else:
                        fw = self.tables.b_pawn_fwd1[sq].item()
                        ct = self.tables.b_pawn_caps
                    # Forward move (only to empty)
                    if fw >= 0 and bds[mi, fw].item() == EMPTY:
                        tgts[fw] = True
                    # Capture diagonals
                    for ci in range(2):
                        t = ct[sq, ci].item()
                        if t >= 0:
                            tgts[t] = True
                else:
                    continue

                # Filter targets: only allow empty squares and controller's own pieces
                # Block moves to pieces of the parried piece's own color (= enemy from controller's POV... 
                # but actually these are the parried piece's teammates, and you can't eat your own teammates)
                for t_sq in tgts.nonzero(as_tuple=True)[0]:
                    t_sq_i = t_sq.item()
                    target_piece = bds[mi, t_sq_i].item()

                    if target_piece == EMPTY:
                        # Empty square = good move (always legal)
                        pm[mi, sq, t_sq_i] = True
                    else:
                        target_is_white = (1 <= target_piece <= 6)
                        if target_is_white == controller_is_white:
                            # Target is controller's own piece = self-capture (legal but bad)
                            pm[mi, sq, t_sq_i] = True
                        # else: target is same color as parried piece = illegal (can't eat own teammates)

            # No castling during parry
            # (already not included since we build manually)

            pf = pm.view(M, 4096)

            # King safety filter (don't leave parried piece's king in check)
            pf = self._filter_king_safety_batched(bds, pf, pc_is_w, self.en_passant[idx])

            # Always allow skip (from == to)
            for mi_idx in range(M):
                sq = p_sq[mi_idx].item()
                skip_action = sq * 64 + sq
                pf[mi_idx, skip_action] = True

            mask[in_parry, :4096] = pf

        return mask

    # ══════════════════════════════════════════════
    #  STEP
    # ══════════════════════════════════════════════

    def step(self, actions):
        N, dev = self.N, self.device
        reward = torch.full((N,), REWARD_STEP_PENALTY, device=dev)

        in_def = self.phase == PHASE_DEFENSE
        if in_def.any(): self._resolve_defense(actions, in_def, reward)
        in_move = self.phase == PHASE_MOVE
        if in_move.any(): self._apply_board_moves(actions, in_move, reward, False)
        in_parry = self.phase == PHASE_PARRY
        if in_parry.any(): self._apply_board_moves(actions, in_parry, reward, True)

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

    def _resolve_defense(self, actions, mask, reward):
        for idx in mask.nonzero(as_tuple=True)[0]:
            i = idx.item(); a = actions[i].item()
            ap = self.pending_attacker_piece[i].item()
            dp = self.pending_defender_piece[i].item()
            at = (ap-1) if ap<=6 else (ap-7); dt = (dp-1) if dp<=6 else (dp-7)
            atk_stat = self.tables.attack_stats[at].item()
            def_stat = self.tables.defense_stats[dt].item()
            tau = def_stat / (atk_stat + def_stat)

            if a==4162: out="E"
            elif a==4160: out="B" if torch.rand(1,device=self.device).item()<tau else "E"
            elif a==4161: out="P" if torch.rand(1,device=self.device).item()<tau else "E"
            else: out="E"

            fs=self.pending_attacker_sq[i].item(); ts=self.pending_target_sq[i].item()
            aw=self.pending_attacker_color_white[i].item()
            agent_is_attacker = (aw == self.agent_is_white[i].item())

            if out=="E":
                self.board[i,ts]=self.board[i,fs]; self.board[i,fs]=EMPTY
                self._post_move_updates(i,fs,ts)
                self.turn_is_white[i]=not aw; self.phase[i]=PHASE_MOVE
                dv=self.tables.piece_values[dt].item()

                if agent_is_attacker:
                    reward[i] += REWARD_CAPTURE_SCALE * dv
                    if atk_stat > def_stat:
                        reward[i] += REWARD_CAPTURE_ATTACKER_BONUS * (atk_stat - def_stat)
                        self.capture_high_attacker_count += 1
                    self.capture_total_count += 1
                else:
                    reward[i] += REWARD_LOSE_PIECE_SCALE * dv

                if a==4162:
                    if not agent_is_attacker:
                        reward[i] += REWARD_ACCEPT_LOSS
                else:
                    if not agent_is_attacker:
                        reward[i] += REWARD_DEFENSE_FAIL

                self.half_moves[i] = 0

            elif out=="B":
                self.turn_is_white[i]=not aw; self.phase[i]=PHASE_MOVE
                if not agent_is_attacker:
                    reward[i] += REWARD_BLOCK_SUCCESS
                else:
                    reward[i] -= REWARD_BLOCK_SUCCESS * 0.5

            elif out=="P":
                self.turn_is_white[i]=not aw; self.phase[i]=PHASE_PARRY
                self.parry_square[i]=fs; self.parry_controller_is_white[i]=not aw
                if not agent_is_attacker:
                    reward[i] += REWARD_PARRY_SUCCESS
                else:
                    reward[i] -= REWARD_PARRY_SUCCESS * 0.5

            self.pending_attacker_sq[i]=-1; self.pending_target_sq[i]=-1
            self._enforce_check(i, aw, reward)

    def _apply_board_moves(self, actions, mask, reward, parry):
        for idx in mask.nonzero(as_tuple=True)[0]:
            i=idx.item(); a=actions[i].item()
            if a>=4096: continue
            fs=a//64; ts=a%64; bd=self.board[i]
            mv=bd[fs].item()
            if mv==EMPTY: continue
            tgt=bd[ts].item(); ic=tgt!=EMPTY; mw=1<=mv<=6
            pt=(mv-1) if mv<=6 else (mv-7)
            ie=(pt==0 and ts==self.en_passant[i].item() and tgt==EMPTY)
            if ie:
                ic=True; cs=(ts%8)+(fs//8)*8; tgt=bd[cs].item(); bd[cs]=EMPTY

            if parry:
                cw=self.parry_controller_is_white[i].item()

                # ── SKIP ──
                if fs == ts:
                    self.parry_skip_count += 1
                    self.parry_total_count += 1
                    self.phase[i]=PHASE_MOVE; self.parry_square[i]=-1
                    reward[i]+=REWARD_PARRY_SKIP
                    self._enforce_check(i,cw,reward); continue

                if ic:
                    # v5 FIX: During parry, the only capture possible is self-capture
                    # (controller's piece gets eaten). Enemy capture is blocked at the
                    # legal mask level, but if it somehow gets here, treat as self-capture.
                    tw=1<=tgt<=6
                    # This should always be self-capture (tw == cw) since enemy capture
                    # is filtered out in get_legal_mask. But handle both for safety.
                    self.parry_self_capture_count += 1
                    self.parry_total_count += 1
                    dt_val = (tgt-1) if tgt<=6 else (tgt-7)
                    piece_val = self.tables.piece_values[dt_val].item()
                    bd[ts]=bd[fs]; bd[fs]=EMPTY; self._post_move_updates(i,fs,ts)
                    self.phase[i]=PHASE_MOVE; self.parry_square[i]=-1
                    reward[i] += REWARD_PARRY_SELF_CAPTURE * piece_val
                    self.half_moves[i]=0
                    self._enforce_check(i,cw,reward); continue

                else:
                    # ── GOOD MOVE (empty square) ──
                    self.parry_good_move_count += 1
                    self.parry_total_count += 1
                    bd[ts]=bd[fs]; bd[fs]=EMPTY; self._post_move_updates(i,fs,ts)
                    self.phase[i]=PHASE_MOVE; self.parry_square[i]=-1
                    reward[i]+=REWARD_PARRY_MOVE_GOOD
                    self._enforce_check(i,cw,reward); continue

            # ── Normal move phase (not parry) ──

            # ── Check escape shaping ──
            actor_ci = 0 if mw else 1
            current_check_attempts = self.check_attempts[i, actor_ci].item()
            is_in_check = current_check_attempts > 0
            agent_is_actor = (mw == self.agent_is_white[i].item())

            # v6: counters kept for logging, but no action-type reward here.
            # The escape reward is outcome-based and given in _enforce_check when
            # check_attempts resets to 0 (confirmed escape). This removes the bias
            # that was penalising valid captures (-0.08) and caused escape_by_move_rate
            # to drop to 5%.
            if is_in_check and current_check_attempts >= 2:
                if ic:
                    self.check_3rd_attempt_capture_count += 1
                else:
                    self.check_3rd_attempt_move_count += 1

            if ic:
                if is_in_check:
                    self.check_escape_by_capture_count += 1
                self._start_defense(i,fs,ts,mv,tgt,mw)
            else:
                if is_in_check:
                    self.check_escape_by_move_count += 1

                self.en_passant[i]=-1
                if pt==5:
                    for (cw_c,f,t,rs,rd) in [(True,4,6,7,5),(True,4,2,0,3),(False,60,62,63,61),(False,60,58,56,59)]:
                        if mw==cw_c and fs==f and ts==t:
                            bd[rd]=bd[rs]; bd[rs]=EMPTY
                bd[ts]=bd[fs]; bd[fs]=EMPTY
                if pt==0 and abs(fs//8-ts//8)==2:
                    self.en_passant[i]=(fs%8)+((fs//8+ts//8)//2)*8

                if pt == 0:
                    self.half_moves[i] = 0
                else:
                    self.half_moves[i] += 1

                self._post_move_updates(i,fs,ts); self.turn_is_white[i]=not mw
                self._enforce_check(i, mw, reward)

    def _start_defense(self, i, fs, ts, ap, dp, aw):
        self.phase[i]=PHASE_DEFENSE; self.pending_attacker_sq[i]=fs
        self.pending_target_sq[i]=ts; self.pending_attacker_piece[i]=ap
        self.pending_defender_piece[i]=dp; self.pending_attacker_color_white[i]=aw
        self.turn_is_white[i]=not aw

    def _post_move_updates(self, i, fs, ts):
        bd=self.board[i]; p=bd[ts].item()
        if p==EMPTY: return
        pt=(p-1) if p<=6 else (p-7); iw=1<=p<=6
        if pt==5:
            if iw: self.castling[i,0]=False; self.castling[i,1]=False
            else: self.castling[i,2]=False; self.castling[i,3]=False
        for sq,ci in [(0,1),(7,0),(56,3),(63,2)]:
            if fs==sq and pt==3: self.castling[i,ci]=False
            if ts==sq: self.castling[i,ci]=False
        if pt==0 and ts//8==(7 if iw else 0):
            bd[ts]=W_QUEEN if iw else B_QUEEN

    def _enforce_check(self, i, actor_w, reward):
        actor_in_check = self._is_in_check_batched(
            self.board[i:i+1],
            torch.tensor([actor_w], dtype=torch.bool, device=self.device)
        )[0].item()

        actor_ci = 0 if actor_w else 1

        if actor_in_check:
            self.check_attempts[i, actor_ci] += 1
            current = self.check_attempts[i, actor_ci].item()
            
            agent_is_checker = (not actor_w == self.agent_is_white[i].item())
            
            agent_is_actor = (actor_w == self.agent_is_white[i].item())
            if agent_is_actor:
                reward[i] += REWARD_CHECK_ATTEMPT_PENALTY
            else:
                reward[i] += REWARD_CHECK_GIVEN
                if current >= 2:
                    reward[i] += REWARD_CHECK_2ND_ATTEMPT

            self.turn_is_white[i] = actor_w
            self.phase[i] = PHASE_MOVE
            self.pending_attacker_sq[i] = -1
            self.pending_target_sq[i] = -1
            self.parry_square[i] = -1

            if self.check_attempts[i, actor_ci] >= 3:
                self.phase[i] = PHASE_FINISHED
                self.result[i] = RESULT_BLACK_WIN if actor_w else RESULT_WHITE_WIN
        else:
            # v6: outcome-based check escape reward.
            # Given only when the agent was in check and successfully escaped.
            # Scales with urgency: more consecutive checks = more reward for escaping.
            old_attempts = self.check_attempts[i, actor_ci].item()
            self.check_attempts[i, actor_ci] = 0
            if old_attempts > 0 and (actor_w == self.agent_is_white[i].item()):
                urgency = min(old_attempts, 2)
                reward[i] += REWARD_CHECK_ESCAPE_SUCCESS * urgency

    def _check_endgame(self, reward):
        active = (self.phase==PHASE_MOVE)
        if not active.any(): return
        bds=self.board[active]; ai=active.nonzero(as_tuple=True)[0]
        wm=~(bds==W_KING).any(dim=1); bm=~(bds==B_KING).any(dim=1)
        if wm.any(): self.phase[ai[wm]]=PHASE_FINISHED; self.result[ai[wm]]=RESULT_BLACK_WIN
        if bm.any(): self.phase[ai[bm]]=PHASE_FINISHED; self.result[ai[bm]]=RESULT_WHITE_WIN
        sa=(self.phase==PHASE_MOVE)
        if not sa.any(): return
        si=sa.nonzero(as_tuple=True)[0]
        pseudo=self._gen_pseudo_legal_batched(self.board[si],self.turn_is_white[si],self.en_passant[si],self.castling[si])
        legal=self._filter_king_safety_batched(self.board[si],pseudo,self.turn_is_white[si],self.en_passant[si])
        nm=~legal.any(dim=1)
        if nm.any():
            ni=si[nm]; ic=self._is_in_check_batched(self.board[ni],self.turn_is_white[ni])
            self.phase[ni]=PHASE_FINISHED; iw=self.turn_is_white[ni]
            self.result[ni]=torch.where(ic,
                torch.where(iw, torch.tensor(RESULT_BLACK_WIN,dtype=torch.int8,device=self.device),
                                 torch.tensor(RESULT_WHITE_WIN,dtype=torch.int8,device=self.device)),
                torch.tensor(RESULT_DRAW,dtype=torch.int8,device=self.device))