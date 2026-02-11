#!/usr/bin/env python3
"""
replay_games.py — Replay shortest and longest games with visual board display.
"""
import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from env.chess_obscur_env import ChessObscurEnv, PHASE_MOVE, PHASE_DEFENSE, PHASE_PARRY
from model.network import ChessObscurNetwork
from utils.checkpoint import find_latest_checkpoint


# Piece symbols for display
PIECE_SYMBOLS = {
    0: '·',   # Empty
    1: '♙',   # White Pawn
    2: '♘',   # White Knight
    3: '♗',   # White Bishop
    4: '♖',   # White Rook
    5: '♕',   # White Queen
    6: '♔',   # White King
    -1: '♟',  # Black Pawn
    -2: '♞',  # Black Knight
    -3: '♝',  # Black Bishop
    -4: '♜',  # Black Rook
    -5: '♛',  # Black Queen
    -6: '♚',  # Black King
}

PHASE_NAMES = {
    0: "MOVE",
    1: "DEFENSE",
    2: "PARRY",
    3: "FINISHED"
}


def display_board(board_state, phase, move_num, info=""):
    """Display an 8x8 chess board in the terminal."""
    os.system('clear')

    print(f"\n{'='*50}")
    print(f"  Move #{move_num}  |  Phase: {PHASE_NAMES.get(phase, 'UNKNOWN')}")
    if info:
        print(f"  {info}")
    print(f"{'='*50}\n")

    print("    a  b  c  d  e  f  g  h")
    print("  ┌" + "─"*23 + "┐")

    for rank in range(7, -1, -1):  # 8 to 1
        print(f"{rank+1} │", end="")
        for file in range(8):  # a to h
            square_idx = rank * 8 + file
            piece = board_state[square_idx]
            symbol = PIECE_SYMBOLS.get(piece, '?')
            print(f" {symbol} ", end="")
        print(f"│ {rank+1}")

    print("  └" + "─"*23 + "┘")
    print("    a  b  c  d  e  f  g  h\n")


def convert_board_to_display(board_tensor):
    """Convert board tensor to display format with proper piece encoding."""
    # board_tensor has pieces 0-12: 0=empty, 1-6=white, 7-12=black
    display_board = []
    for piece in board_tensor:
        p = piece.item()
        if p == 0:
            display_board.append(0)
        elif 1 <= p <= 6:
            display_board.append(p)  # White pieces
        elif 7 <= p <= 12:
            display_board.append(-(p - 6))  # Black pieces
        else:
            display_board.append(0)
    return display_board


def load_model(checkpoint_path, cfg, device):
    """Load a model from checkpoint."""
    network = ChessObscurNetwork(
        obs_planes=cfg.obs_planes,
        num_filters=cfg.num_filters,
        num_res_blocks=cfg.num_res_blocks,
        policy_head_filters=cfg.policy_head_filters,
        value_head_hidden=cfg.value_head_hidden,
        total_actions=cfg.total_actions,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Strip _orig_mod prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_state_dict[key.replace("_orig_mod.", "")] = value
        else:
            new_state_dict[key] = value

    network.load_state_dict(new_state_dict)
    network.eval()

    return network


def record_games(network, cfg, num_games=100, device="cuda"):
    """Record games and return the shortest and longest."""
    env = ChessObscurEnv(num_games, device=device, max_steps=300)
    obs = env.reset()

    # Store game histories
    game_histories = [[] for _ in range(num_games)]
    game_lengths = [0] * num_games
    games_done = [False] * num_games

    with torch.no_grad():
        step = 0
        while not all(games_done):
            # Record current board state for all active games
            for i in range(num_games):
                if not games_done[i]:
                    board_copy = env.board[i].cpu().clone()
                    phase_copy = env.phase[i].item()
                    game_histories[i].append((board_copy, phase_copy))

            legal_mask = env.get_legal_mask()
            no_legal = ~legal_mask.any(dim=1)
            if no_legal.any():
                legal_mask[no_legal, 4162] = True

            policy_logits, value = network(obs, legal_mask)
            actions = policy_logits.argmax(dim=-1)

            obs, reward, done, info = env.step(actions)

            if done.any():
                for i in done.nonzero(as_tuple=True)[0]:
                    idx = i.item()
                    if not games_done[idx]:
                        game_lengths[idx] = len(game_histories[idx])
                        games_done[idx] = True

            step += 1
            if step > 2000:  # Safety
                break

    # Find shortest and longest games
    completed_games = [(i, length) for i, length in enumerate(game_lengths) if length > 0]
    if not completed_games:
        return None, None

    shortest_idx = min(completed_games, key=lambda x: x[1])[0]
    longest_idx = max(completed_games, key=lambda x: x[1])[0]

    return game_histories[shortest_idx], game_histories[longest_idx]


def replay_game(game_history, title, delay=0.5):
    """Replay a game with visual display."""
    print(f"\n\n{'='*50}")
    print(f"  {title}")
    print(f"  Total moves: {len(game_history)}")
    print(f"{'='*50}")
    input("\nPress ENTER to start replay...")

    for move_num, (board, phase) in enumerate(game_history):
        display_board_data = convert_board_to_display(board)
        display_board(display_board_data, phase, move_num + 1)
        time.sleep(delay)

    print("\n  Game finished!")
    input("\nPress ENTER to continue...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Replay Chess Obscur games")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between moves (seconds)")
    parser.add_argument("--value-head-hidden", type=int, default=256)
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.value_head_hidden = args.value_head_hidden

    if args.checkpoint == "latest":
        checkpoint_path = find_latest_checkpoint("checkpoints")
    else:
        checkpoint_path = args.checkpoint

    print(f"\n[replay] Loading model from {checkpoint_path}...")
    network = load_model(checkpoint_path, cfg, args.device)

    print(f"\n[replay] Recording {args.num_games} games...")
    shortest_game, longest_game = record_games(network, cfg, args.num_games, args.device)

    if shortest_game is None:
        print("[error] No games completed!")
        return

    print(f"\n[replay] Found shortest game: {len(shortest_game)} moves")
    print(f"[replay] Found longest game: {len(longest_game)} moves")

    # Replay shortest
    replay_game(shortest_game, "🎮 SHORTEST GAME", delay=args.delay)

    # Replay longest
    replay_game(longest_game, "🎮 LONGEST GAME", delay=args.delay)

    print("\n✅ Replay complete!\n")


if __name__ == "__main__":
    main()
