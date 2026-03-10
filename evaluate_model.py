#!/usr/bin/env python3
"""
evaluate_model.py — Comprehensive evaluation of Chess Obscur RL models.

CHANGES v5:
- Removed parry/enemy_capture tracking (illegal move removed)
- Parry stats: skip, good_move, self_capture only
"""
import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from env.chess_obscur_env import (
    ChessObscurEnv,
    PHASE_MOVE,
    PHASE_DEFENSE,
    PHASE_PARRY,
    PHASE_FINISHED,
    RESULT_ONGOING,
    RESULT_WHITE_WIN,
    RESULT_BLACK_WIN,
    RESULT_DRAW,
    ACTION_ATTEMPT_BLOCK,
    ACTION_ATTEMPT_PARRY,
    ACTION_ACCEPT_LOSS,
)
from model.network import ChessObscurNetwork
from utils.checkpoint import load_checkpoint, find_latest_checkpoint


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

    ckpt = load_checkpoint(checkpoint_path, network, device=device)
    network.eval()

    step = ckpt.get("global_step", 0) if isinstance(ckpt, dict) else 0
    print(f"  Loaded: {checkpoint_path} (step {step:,})")
    return network, step


def run_evaluation(network, cfg, num_games=500, num_envs=256, device="cuda",
                   temperature=0.0, detailed=False):
    num_envs = min(num_envs, num_games)
    env = ChessObscurEnv(num_envs, device=device, max_steps=300)
    obs = env.reset()
    
    stats = {
        "games_completed": 0,
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "game_lengths": [],
        "defense_block": 0,
        "defense_parry": 0,
        "defense_accept": 0,
        "defense_total": 0,
        # Parry: 3 outcomes only
        "parry_skip": 0,
        "parry_self_capture": 0,
        "parry_good_move": 0,
        "parry_total": 0,
        "final_material_advantage": [],
        "steps_move": 0,
        "steps_defense": 0,
        "steps_parry": 0,
        "total_steps": 0,
    }
    
    step_count = 0
    max_steps_per_game = 600 * 2
    
    with torch.no_grad():
        while stats["games_completed"] < num_games:
            legal_mask = env.get_legal_mask()
            
            stats["steps_move"] += (env.phase == PHASE_MOVE).sum().item()
            stats["steps_defense"] += (env.phase == PHASE_DEFENSE).sum().item()
            stats["steps_parry"] += (env.phase == PHASE_PARRY).sum().item()
            stats["total_steps"] += num_envs
            
            no_legal = ~legal_mask.any(dim=1)
            if no_legal.any():
                legal_mask[no_legal, ACTION_ACCEPT_LOSS] = True
            
            policy_logits, value = network(obs, legal_mask)
            
            if temperature > 0:
                policy_logits = policy_logits / temperature
                policy_logits = policy_logits.masked_fill(~legal_mask, -1e8)
                probs = F.softmax(policy_logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
            else:
                actions = policy_logits.argmax(dim=-1)
            
            # Track defense decisions
            in_def = env.phase == PHASE_DEFENSE
            if in_def.any():
                def_actions = actions[in_def]
                stats["defense_total"] += def_actions.shape[0]
                stats["defense_block"] += (def_actions == ACTION_ATTEMPT_BLOCK).sum().item()
                stats["defense_parry"] += (def_actions == ACTION_ATTEMPT_PARRY).sum().item()
                stats["defense_accept"] += (def_actions == ACTION_ACCEPT_LOSS).sum().item()
            
            # Track parry decisions (3 outcomes)
            in_parry = env.phase == PHASE_PARRY
            if in_parry.any():
                parry_actions = actions[in_parry]
                parry_indices = in_parry.nonzero(as_tuple=True)[0]
                
                for pi, idx in enumerate(parry_indices):
                    a = parry_actions[pi].item()
                    stats["parry_total"] += 1
                    
                    if a >= 4096:
                        continue
                    
                    fs = a // 64
                    ts = a % 64
                    
                    if fs == ts:
                        stats["parry_skip"] += 1
                    else:
                        i = idx.item()
                        target_piece = env.board[i, ts].item()
                        if target_piece == 0:
                            stats["parry_good_move"] += 1
                        else:
                            # Any capture during parry is self-capture
                            stats["parry_self_capture"] += 1
            
            obs, reward, done, info = env.step(actions)
            step_count += 1
            
            if done.any():
                for i in done.nonzero(as_tuple=True)[0]:
                    result = info["result"][i].item()
                    if result == RESULT_WHITE_WIN:
                        stats["white_wins"] += 1
                    elif result == RESULT_BLACK_WIN:
                        stats["black_wins"] += 1
                    elif result == RESULT_DRAW:
                        stats["draws"] += 1
                    
                    stats["games_completed"] += 1
                    stats["game_lengths"].append(info["full_move_count"][i].item())
                    
                    if stats["games_completed"] >= num_games:
                        break
            
            if step_count > num_games * max_steps_per_game // num_envs:
                print(f"  [warn] Hit step limit at {step_count} steps, {stats['games_completed']} games done")
                break
    
    return stats


def print_stats(stats, label="Evaluation"):
    total = stats["games_completed"]
    if total == 0:
        print(f"\n{label}: No games completed!")
        return
    
    print(f"\n{'=' * 70}")
    print(f"  {label} — {total} games")
    print(f"{'=' * 70}")
    
    ww = stats["white_wins"]
    bw = stats["black_wins"]
    dr = stats["draws"]
    print(f"\n  Results:")
    print(f"    White wins:  {ww:>5} ({ww/total*100:5.1f}%)")
    print(f"    Black wins:  {bw:>5} ({bw/total*100:5.1f}%)")
    print(f"    Draws:       {dr:>5} ({dr/total*100:5.1f}%)")
    print(f"    Win rate:    {(ww+bw)/total*100:5.1f}%")
    
    lengths = stats["game_lengths"]
    if lengths:
        print(f"\n  Game Length:")
        print(f"    Mean:    {sum(lengths)/len(lengths):6.1f} half-moves")
        print(f"    Median:  {sorted(lengths)[len(lengths)//2]:6.1f}")
        print(f"    Min:     {min(lengths):6.1f}")
        print(f"    Max:     {max(lengths):6.1f}")
    
    ts = stats["total_steps"]
    if ts > 0:
        print(f"\n  Phase Distribution:")
        print(f"    Move:    {stats['steps_move']/ts*100:5.1f}%")
        print(f"    Defense: {stats['steps_defense']/ts*100:5.1f}%")
        print(f"    Parry:   {stats['steps_parry']/ts*100:5.1f}%")
    
    dt = stats["defense_total"]
    if dt > 0:
        print(f"\n  Defense Decisions ({dt} total):")
        print(f"    Block:       {stats['defense_block']:>5} ({stats['defense_block']/dt*100:5.1f}%)")
        print(f"    Parry:       {stats['defense_parry']:>5} ({stats['defense_parry']/dt*100:5.1f}%)")
        print(f"    Accept loss: {stats['defense_accept']:>5} ({stats['defense_accept']/dt*100:5.1f}%)")
    
    pt = stats["parry_total"]
    if pt > 0:
        print(f"\n  Parry Decisions ({pt} total):")
        print(f"    Skip (no move):    {stats['parry_skip']:>5} ({stats['parry_skip']/pt*100:5.1f}%)")
        print(f"    Good move (empty): {stats['parry_good_move']:>5} ({stats['parry_good_move']/pt*100:5.1f}%)")
        print(f"    SELF-CAPTURE:      {stats['parry_self_capture']:>5} ({stats['parry_self_capture']/pt*100:5.1f}%)")
        
        if stats['parry_self_capture'] / pt > 0.15:
            print(f"\n    WARNING: SELF-CAPTURE RATE IS HIGH ({stats['parry_self_capture']/pt*100:.1f}%)")
        elif stats['parry_self_capture'] / pt < 0.05:
            print(f"\n    OK: Self-capture rate is low ({stats['parry_self_capture']/pt*100:.1f}%)")
    
    print(f"\n{'=' * 70}")


def compare_models(stats_a, stats_b, label_a="Model A", label_b="Model B"):
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'=' * 70}")
    
    ta = stats_a["games_completed"]
    tb = stats_b["games_completed"]
    
    if ta == 0 or tb == 0:
        print("  Cannot compare — one model has no games.")
        return
    
    metrics = [
        ("Win rate", lambda s: (s["white_wins"]+s["black_wins"])/s["games_completed"]*100, "%"),
        ("Draw rate", lambda s: s["draws"]/s["games_completed"]*100, "%"),
        ("Avg length", lambda s: sum(s["game_lengths"])/max(len(s["game_lengths"]),1), " moves"),
        ("White WR", lambda s: s["white_wins"]/s["games_completed"]*100, "%"),
        ("Black WR", lambda s: s["black_wins"]/s["games_completed"]*100, "%"),
    ]
    
    if stats_a["defense_total"] > 0:
        metrics.append(("Block %", lambda s: s["defense_block"]/max(s["defense_total"],1)*100, "%"))
        metrics.append(("Parry %", lambda s: s["defense_parry"]/max(s["defense_total"],1)*100, "%"))
    
    if stats_a["parry_total"] > 0:
        metrics.append(("Parry skip %", lambda s: s["parry_skip"]/max(s["parry_total"],1)*100, "%"))
        metrics.append(("Self-capture %", lambda s: s["parry_self_capture"]/max(s["parry_total"],1)*100, "%"))
        metrics.append(("Good parry %", lambda s: s["parry_good_move"]/max(s["parry_total"],1)*100, "%"))
    
    print(f"\n  {'Metric':<20} {label_a:>12} {label_b:>12}   {'Delta':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}   {'-'*10}")
    
    for name, fn, unit in metrics:
        va = fn(stats_a)
        vb = fn(stats_b)
        delta = vb - va
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        print(f"  {name:<20} {va:>10.1f}{unit} {vb:>10.1f}{unit}   {arrow} {abs(delta):>7.1f}{unit}")
    
    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Chess Obscur Model Evaluation")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-a", default=None)
    parser.add_argument("--checkpoint-b", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parry-analysis", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--value-head-hidden", type=int, default=None)
    parser.add_argument("--num-res-blocks", type=int, default=None)
    parser.add_argument("--num-filters", type=int, default=None)
    args = parser.parse_args()
    
    cfg = Config()
    cfg.device = args.device
    if args.value_head_hidden is not None:
        cfg.value_head_hidden = args.value_head_hidden
    if args.num_res_blocks is not None:
        cfg.num_res_blocks = args.num_res_blocks
    if args.num_filters is not None:
        cfg.num_filters = args.num_filters
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, using CPU")
        args.device = "cpu"
        cfg.device = "cpu"
    
    if args.checkpoint_a and args.checkpoint_b:
        print(f"\n[eval] Comparing two models with {args.num_games} games each")
        
        print(f"\n[eval] Loading Model A...")
        net_a, step_a = load_model(args.checkpoint_a, cfg, args.device)
        
        print(f"\n[eval] Loading Model B...")
        net_b, step_b = load_model(args.checkpoint_b, cfg, args.device)
        
        print(f"\n[eval] Evaluating Model A (step {step_a:,})...")
        t0 = time.time()
        stats_a = run_evaluation(net_a, cfg, args.num_games, args.num_envs, args.device,
                                  args.temperature, detailed=args.parry_analysis)
        print(f"  Done in {time.time()-t0:.1f}s")
        
        print(f"\n[eval] Evaluating Model B (step {step_b:,})...")
        t0 = time.time()
        stats_b = run_evaluation(net_b, cfg, args.num_games, args.num_envs, args.device,
                                  args.temperature, detailed=args.parry_analysis)
        print(f"  Done in {time.time()-t0:.1f}s")
        
        label_a = f"Step {step_a:,}"
        label_b = f"Step {step_b:,}"
        
        print_stats(stats_a, label=label_a)
        print_stats(stats_b, label=label_b)
        compare_models(stats_a, stats_b, label_a, label_b)
    
    else:
        ckpt = args.checkpoint
        if ckpt == "latest" or ckpt is None:
            ckpt = find_latest_checkpoint(args.checkpoint_dir)
        
        if not ckpt or not os.path.exists(ckpt):
            print(f"[error] No checkpoint found: {ckpt}")
            return
        
        print(f"\n[eval] Loading model...")
        network, step = load_model(ckpt, cfg, args.device)
        
        print(f"\n[eval] Running {args.num_games} self-play games (temp={args.temperature})...")
        t0 = time.time()
        stats = run_evaluation(network, cfg, args.num_games, args.num_envs, args.device,
                                args.temperature, detailed=args.parry_analysis)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({stats['games_completed']/elapsed:.0f} games/sec)")
        
        print_stats(stats, label=f"Step {step:,}")


if __name__ == "__main__":
    main()
