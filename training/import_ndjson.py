"""
import_ndjson.py — Import games from Chess Obscur website for behavioral cloning warmstart.

Reads the NDJSON dataset exported by the website (/api/dataset.ndjson) and
converts it into (observation, action, result) tensors that can be used to
warmstart the PPO network via supervised learning.

Usage:
    python -m training.import_ndjson --file dataset.ndjson --output data/human_games.pt
"""
import json
import argparse
import torch
from typing import List, Dict, Tuple, Optional

from env.move_tables import (
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    EMPTY, file_rank_to_idx,
)

# Piece code mapping from chess.js notation to our int encoding
PIECE_MAP = {
    "P": W_PAWN, "N": W_KNIGHT, "B": W_BISHOP, "R": W_ROOK, "Q": W_QUEEN, "K": W_KING,
    "p": B_PAWN, "n": B_KNIGHT, "b": B_BISHOP, "r": B_ROOK, "q": B_QUEEN, "k": B_KING,
}

# Move types from the website that represent actual board moves
BOARD_MOVE_TYPES = {"MOVE", "BOT_MOVE", "CAPTURE_SUCCESS", "PARRY_MOVE", "PARRY_SELF_CAPTURE"}
DEFENSE_TYPES = {"DEFENSE_RESOLVE"}

# Result mapping
RESULT_MAP = {
    "WHITE_WIN_KING_CAPTURED": 1, "WHITE_WIN_CHECKMATE": 1,
    "WHITE_WIN_FORCED_CHECK": 1, "WHITE_WIN_PARRY_KING_FORCED_CHECK": 1,
    "BLACK_WIN_KING_CAPTURED": 2, "BLACK_WIN_CHECKMATE": 2,
    "BLACK_WIN_FORCED_CHECK": 2, "BLACK_WIN_PARRY_KING_FORCED_CHECK": 2,
    "DRAW_STALEMATE": 3,
}


def sq_to_idx(sq: str) -> int:
    """Convert algebraic notation to board index (e.g. 'e2' -> 12)."""
    f = ord(sq[0]) - ord('a')
    r = int(sq[1]) - 1
    return f + r * 8


def board_from_js(js_board: list) -> torch.Tensor:
    """Convert chess.js board array (64 elements, null or piece string) to our tensor."""
    board = torch.zeros(64, dtype=torch.int8)
    for i, p in enumerate(js_board):
        if p is not None and p in PIECE_MAP:
            board[i] = PIECE_MAP[p]
    return board


def build_obs_from_board(board: torch.Tensor, is_white_turn: bool,
                         castling: dict = None, en_passant: int = -1,
                         phase: int = 0, check_attempts: int = 0,
                         half_moves: int = 0) -> torch.Tensor:
    """Build a (19, 8, 8) observation tensor from raw state."""
    obs = torch.zeros(19, 8, 8)

    for pt in range(6):
        w_code = pt + 1
        b_code = pt + 7
        w_mask = (board == w_code).float().view(8, 8)
        b_mask = (board == b_code).float().view(8, 8)

        if is_white_turn:
            obs[pt] = w_mask
            obs[6 + pt] = b_mask
        else:
            obs[pt] = b_mask
            obs[6 + pt] = w_mask

    # En passant
    if 0 <= en_passant < 64:
        r, f = en_passant // 8, en_passant % 8
        obs[12, r, f] = 1.0

    # Castling
    if castling:
        wK = float(castling.get("wK", False))
        wQ = float(castling.get("wQ", False))
        bK = float(castling.get("bK", False))
        bQ = float(castling.get("bQ", False))
        if is_white_turn:
            obs[13, 0, :] = wK; obs[13, 1, :] = wQ
            obs[13, 2, :] = bK; obs[13, 3, :] = bQ
        else:
            obs[13, 0, :] = bK; obs[13, 1, :] = bQ
            obs[13, 2, :] = wK; obs[13, 3, :] = wQ

    # Turn
    obs[14] = 1.0 if is_white_turn else 0.0

    # Phase, check attempts, half-move
    obs[16] = phase / 3.0
    obs[17] = check_attempts / 3.0
    obs[18] = min(half_moves / 100.0, 1.0)

    return obs


def move_to_action(move: dict) -> Optional[int]:
    """Convert a website move event to our action index."""
    move_type = move.get("type", "")

    if move_type in BOARD_MOVE_TYPES:
        from_sq = move.get("from", "")
        to_sq = move.get("to", "")
        if not from_sq or not to_sq:
            return None
        fr = sq_to_idx(from_sq)
        to = sq_to_idx(to_sq)
        return fr * 64 + to

    if move_type == "DEFENSE_RESOLVE":
        attempt = move.get("attempt", "")
        if attempt == "BLOCAGE":
            return 4160  # ATTEMPT_BLOCK
        elif attempt == "PARADE":
            return 4161  # ATTEMPT_PARRY
        elif attempt in ("ACCEPT_LOSS", "PASSE", "ECHEC_ZONE"):
            return 4162  # ACCEPT_LOSS
        return 4162  # default

    return None


def load_ndjson_games(filepath: str) -> Dict[str, List[dict]]:
    """Load NDJSON and group moves by gameId."""
    games = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                gid = record.get("gameId", "unknown")
                if gid not in games:
                    games[gid] = []
                games[gid].append(record)
            except json.JSONDecodeError:
                continue
    return games


def replay_game(moves_records: List[dict]) -> List[Tuple[torch.Tensor, int, int]]:
    """
    Replay a game from NDJSON records and extract (obs, action, result) tuples.
    
    We reconstruct the board state step by step using the move events,
    and extract training samples at each decision point.
    
    Returns: list of (obs_tensor, action_int, result_int)
    """
    # We need to replay from scratch - the NDJSON doesn't include board snapshots
    # So we maintain a simple board state and replay moves
    
    # Starting board
    board = torch.zeros(64, dtype=torch.int8)
    back_w = [W_ROOK, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING, W_BISHOP, W_KNIGHT, W_ROOK]
    back_b = [B_ROOK, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING, B_BISHOP, B_KNIGHT, B_ROOK]
    for i in range(8):
        board[i] = back_w[i]
        board[8 + i] = W_PAWN
        board[48 + i] = B_PAWN
        board[56 + i] = back_b[i]

    is_white_turn = True
    castling = {"wK": True, "wQ": True, "bK": True, "bQ": True}
    en_passant = -1
    half_moves = 0
    samples = []

    # Determine game result
    result = 0  # ongoing/unknown
    # We don't have the result in NDJSON per-move, but we can infer from the last move type
    
    for record in moves_records:
        move = record.get("move", {})
        move_type = move.get("type", "")
        by = move.get("by", "")

        action = move_to_action(move)
        if action is None:
            continue

        # Build observation BEFORE the move
        phase = 0
        if move_type == "DEFENSE_RESOLVE":
            phase = 1  # defense
        elif move_type in ("PARRY_MOVE", "PARRY_SELF_CAPTURE"):
            phase = 2  # parry

        actor_is_white = (by == "w")
        obs = build_obs_from_board(board, actor_is_white, castling, en_passant,
                                   phase, 0, half_moves)
        samples.append((obs, action, 0))  # result filled later

        # Apply the move to update board state
        if move_type in BOARD_MOVE_TYPES:
            from_sq_str = move.get("from", "")
            to_sq_str = move.get("to", "")
            if from_sq_str and to_sq_str:
                fr = sq_to_idx(from_sq_str)
                to = sq_to_idx(to_sq_str)

                # Handle capture (piece at target gets removed)
                board[to] = board[fr]
                board[fr] = EMPTY

                # En passant reset
                piece_type = board[to].item()
                pt = (piece_type - 1) if piece_type <= 6 else (piece_type - 7)

                en_passant = -1
                if pt == 0 and abs(fr // 8 - to // 8) == 2:
                    mid_rank = (fr // 8 + to // 8) // 2
                    en_passant = (fr % 8) + mid_rank * 8

                # Promotion
                promo = move.get("promotion")
                if promo:
                    promo_map = {"q": W_QUEEN, "r": W_ROOK, "b": W_BISHOP, "n": W_KNIGHT}
                    promo_map_b = {"q": B_QUEEN, "r": B_ROOK, "b": B_BISHOP, "n": B_KNIGHT}
                    if actor_is_white:
                        board[to] = promo_map.get(promo, W_QUEEN)
                    else:
                        board[to] = promo_map_b.get(promo, B_QUEEN)

                # Castling
                if pt == 5:
                    if actor_is_white:
                        castling["wK"] = False; castling["wQ"] = False
                        if fr == 4 and to == 6:
                            board[5] = board[7]; board[7] = EMPTY
                        elif fr == 4 and to == 2:
                            board[3] = board[0]; board[0] = EMPTY
                    else:
                        castling["bK"] = False; castling["bQ"] = False
                        if fr == 60 and to == 62:
                            board[61] = board[63]; board[63] = EMPTY
                        elif fr == 60 and to == 58:
                            board[59] = board[56]; board[56] = EMPTY

                half_moves += 1
                is_white_turn = not actor_is_white

    return samples


def process_ndjson(filepath: str, output_path: str):
    """Process NDJSON file and save training tensors."""
    print(f"[import] Loading {filepath}...")
    games = load_ndjson_games(filepath)
    print(f"[import] Found {len(games)} games")

    all_obs = []
    all_actions = []
    all_results = []

    for gid, records in games.items():
        try:
            samples = replay_game(records)
            for obs, action, result in samples:
                all_obs.append(obs)
                all_actions.append(action)
                all_results.append(result)
        except Exception as e:
            print(f"[import] Error processing game {gid}: {e}")
            continue

    if not all_obs:
        print("[import] No samples extracted!")
        return

    obs_tensor = torch.stack(all_obs)       # (S, 19, 8, 8)
    action_tensor = torch.tensor(all_actions, dtype=torch.int64)  # (S,)
    result_tensor = torch.tensor(all_results, dtype=torch.int8)   # (S,)

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({
        "obs": obs_tensor,
        "actions": action_tensor,
        "results": result_tensor,
    }, output_path)

    print(f"[import] Saved {len(all_obs)} samples to {output_path}")
    print(f"[import] Obs shape: {obs_tensor.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Chess Obscur website games")
    parser.add_argument("--file", required=True, help="Path to dataset.ndjson")
    parser.add_argument("--output", default="data/human_games.pt", help="Output .pt file")
    args = parser.parse_args()

    process_ndjson(args.file, args.output)
