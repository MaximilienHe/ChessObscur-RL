"""
move_tables.py — Precomputed move lookup tables on GPU.

All piece movement patterns are stored as tensors so legal move masks
can be computed with pure tensor operations (no Python loops over squares).

Board indexing: idx = file + rank*8  (file=col 0-7, rank=row 0-7)
  a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63

Piece encoding in board tensor (int8):
  0 = empty
  White: P=1, N=2, B=3, R=4, Q=5, K=6
  Black: p=7, n=8, b=9, r=10, q=11, k=12
"""
import torch
from typing import Tuple

# Piece type indices (color-independent)
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
PIECE_NAMES = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]

# White pieces = 1..6, Black = 7..12
W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING = 1, 2, 3, 4, 5, 6
B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING = 7, 8, 9, 10, 11, 12

EMPTY = 0


def idx_to_file_rank(idx: int) -> Tuple[int, int]:
    return idx % 8, idx // 8


def file_rank_to_idx(f: int, r: int) -> int:
    return f + r * 8


def in_bounds(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def build_knight_moves() -> torch.Tensor:
    offsets = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    table = torch.full((64, 8), -1, dtype=torch.int16)
    for sq in range(64):
        f, r = idx_to_file_rank(sq)
        for i, (df, dr) in enumerate(offsets):
            nf, nr = f + df, r + dr
            if in_bounds(nf, nr):
                table[sq, i] = file_rank_to_idx(nf, nr)
    return table


def build_king_moves() -> torch.Tensor:
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    table = torch.full((64, 8), -1, dtype=torch.int16)
    for sq in range(64):
        f, r = idx_to_file_rank(sq)
        for i, (df, dr) in enumerate(offsets):
            nf, nr = f + df, r + dr
            if in_bounds(nf, nr):
                table[sq, i] = file_rank_to_idx(nf, nr)
    return table


def build_ray_moves() -> torch.Tensor:
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    table = torch.full((64, 8, 7), -1, dtype=torch.int16)
    for sq in range(64):
        f, r = idx_to_file_rank(sq)
        for d, (df, dr) in enumerate(directions):
            nf, nr = f + df, r + dr
            step = 0
            while in_bounds(nf, nr) and step < 7:
                table[sq, d, step] = file_rank_to_idx(nf, nr)
                nf += df
                nr += dr
                step += 1
    return table


def build_pawn_moves_white() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fwd1 = torch.full((64,), -1, dtype=torch.int16)
    fwd2 = torch.full((64,), -1, dtype=torch.int16)
    caps = torch.full((64, 2), -1, dtype=torch.int16)
    for sq in range(64):
        f, r = idx_to_file_rank(sq)
        if r < 7:
            fwd1[sq] = file_rank_to_idx(f, r + 1)
        if r == 1:
            fwd2[sq] = file_rank_to_idx(f, r + 2)
        for ci, df in enumerate([-1, 1]):
            nf = f + df
            nr = r + 1
            if in_bounds(nf, nr):
                caps[sq, ci] = file_rank_to_idx(nf, nr)
    return fwd1, fwd2, caps


def build_pawn_moves_black() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fwd1 = torch.full((64,), -1, dtype=torch.int16)
    fwd2 = torch.full((64,), -1, dtype=torch.int16)
    caps = torch.full((64, 2), -1, dtype=torch.int16)
    for sq in range(64):
        f, r = idx_to_file_rank(sq)
        if r > 0:
            fwd1[sq] = file_rank_to_idx(f, r - 1)
        if r == 6:
            fwd2[sq] = file_rank_to_idx(f, r - 2)
        for ci, df in enumerate([-1, 1]):
            nf = f + df
            nr = r - 1
            if in_bounds(nf, nr):
                caps[sq, ci] = file_rank_to_idx(nf, nr)
    return fwd1, fwd2, caps


class MoveTables:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

        self.knight_moves = build_knight_moves().to(self.device)
        self.king_moves = build_king_moves().to(self.device)
        self.ray_moves = build_ray_moves().to(self.device)

        wf1, wf2, wc = build_pawn_moves_white()
        self.w_pawn_fwd1 = wf1.to(self.device)
        self.w_pawn_fwd2 = wf2.to(self.device)
        self.w_pawn_caps = wc.to(self.device)

        bf1, bf2, bc = build_pawn_moves_black()
        self.b_pawn_fwd1 = bf1.to(self.device)
        self.b_pawn_fwd2 = bf2.to(self.device)
        self.b_pawn_caps = bc.to(self.device)

        self.bishop_dirs = torch.tensor([1, 3, 5, 7], device=self.device)
        self.rook_dirs = torch.tensor([0, 2, 4, 6], device=self.device)
        self.queen_dirs = torch.arange(8, device=self.device)

        self.castle_info = {
            "wK": {"king_from": 4, "king_to": 6, "rook_from": 7, "rook_to": 5,
                    "pass_through": [5, 6], "rook_piece": W_ROOK},
            "wQ": {"king_from": 4, "king_to": 2, "rook_from": 0, "rook_to": 3,
                    "pass_through": [1, 2, 3], "empty_check": [2, 3], "rook_piece": W_ROOK},
            "bK": {"king_from": 60, "king_to": 62, "rook_from": 63, "rook_to": 61,
                    "pass_through": [61, 62], "rook_piece": B_ROOK},
            "bQ": {"king_from": 60, "king_to": 58, "rook_from": 56, "rook_to": 59,
                    "pass_through": [57, 58, 59], "empty_check": [58, 59], "rook_piece": B_ROOK},
        }

        self.attack_stats = torch.tensor([1, 3, 3, 5, 9, 10], dtype=torch.float32, device=self.device)
        self.defense_stats = torch.tensor([1, 3, 3, 7, 8, 10], dtype=torch.float32, device=self.device)
        self.piece_values = torch.tensor([1, 3, 3, 5, 9, 100], dtype=torch.float32, device=self.device)