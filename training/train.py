"""
train.py — Main continuous training loop for Chess Obscur PPO.
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from env.chess_obscur_env import ChessObscurEnv
from model.network import ChessObscurNetwork
from model.ppo import PPOTrainer
from training.self_play import RolloutBuffer, collect_rollout
from utils.checkpoint import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint, checkpoint_path
)
from utils.logger import Logger


# ─────────────────────────────────────────────────────────────
# Behavioral cloning warmstart
# ─────────────────────────────────────────────────────────────

def warmstart_from_human_data(network: ChessObscurNetwork, data_path: str, cfg: Config):
    print(f"\n{'='*60}")
    print(f"[warmstart] Loading human data from {data_path}")
    data = torch.load(data_path, map_location=cfg.device, weights_only=False)

    obs = data["obs"].to(cfg.device)
    actions = data["actions"].to(cfg.device)
    results = data["results"].to(cfg.device)

    S = obs.shape[0]
    print(f"[warmstart] {S} samples loaded")

    if S == 0:
        print("[warmstart] No samples, skipping.")
        return

    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.warmstart_lr)
    network.train()

    batch_size = min(cfg.warmstart_batch, S)

    for epoch in range(cfg.warmstart_epochs):
        perm = torch.randperm(S, device=cfg.device)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0

        for start in range(0, S, batch_size):
            end = min(start + batch_size, S)
            idx = perm[start:end]

            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_results = results[idx]

            policy_logits, value = network(mb_obs)
            policy_loss = F.cross_entropy(policy_logits, mb_actions)

            value_target = torch.zeros_like(mb_results, dtype=torch.float32)
            value_target[mb_results == 1] = 1.0
            value_target[mb_results == 2] = -1.0
            value_loss = F.mse_loss(value.squeeze(-1), value_target)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        avg_p = total_policy_loss / max(n_batches, 1)
        avg_v = total_value_loss / max(n_batches, 1)
        print(f"[warmstart] epoch {epoch+1}/{cfg.warmstart_epochs} | "
              f"policy_loss={avg_p:.4f} | value_loss={avg_v:.4f}")

    print(f"[warmstart] Done.\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(network: ChessObscurNetwork, cfg: Config, num_games: int = 100):
    print(f"\n[eval] Running {num_games} evaluation games...")
    device = cfg.device
    num_envs = min(num_games, cfg.num_envs)
    env = ChessObscurEnv(num_envs, device=device, max_steps=cfg.max_game_steps)

    obs = env.reset()
    network.eval()

    games_done = 0
    results = {1: 0, 2: 0, 3: 0}
    game_lengths = []
    step_count = 0

    with torch.no_grad():
        while games_done < num_games:
            legal_mask = env.get_legal_mask()
            no_legal = ~legal_mask.any(dim=1)
            if no_legal.any():
                legal_mask[no_legal, 4162] = True

            policy_logits, value = network(obs, legal_mask)
            actions = policy_logits.argmax(dim=-1)

            obs, reward, done, info = env.step(actions)
            step_count += 1

            if done.any():
                for i in done.nonzero(as_tuple=True)[0]:
                    r = info["result"][i].item()
                    if r in results:
                        results[r] += 1
                    games_done += 1
                    game_lengths.append(info["full_move_count"][i].item())

            if step_count > num_games * cfg.max_game_steps:
                break

    total = sum(results.values()) or 1
    print(f"\n[eval] Results over {total} games:")
    print(f"  White wins: {results[1]} ({results[1]/total*100:.1f}%)")
    print(f"  Black wins: {results[2]} ({results[2]/total*100:.1f}%)")
    print(f"  Draws:      {results[3]} ({results[3]/total*100:.1f}%)")
    if game_lengths:
        print(f"  Avg game length: {sum(game_lengths)/len(game_lengths):.0f} half-moves")
    print()

    return results


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def train(cfg: Config, resume: str = None, warmstart: str = None):
    device = cfg.device
    print(f"\n{'='*60}")
    print(f"Chess Obscur PPO Training")
    print(f"  Device:     {device}")
    print(f"  Num envs:   {cfg.num_envs}")
    print(f"  Rollout:    {cfg.rollout_steps} steps")
    print(f"  Batch:      {cfg.batch_size}")
    print(f"  Minibatch:  {cfg.minibatch_size}")
    print(f"  Microbatch: {cfg.microbatch_size}")
    print(f"  PPO epochs: {cfg.ppo_epochs}")
    print(f"  Max game steps: {cfg.max_game_steps}")
    print(f"  Total steps: {cfg.total_timesteps:,}")
    print(f"{'='*60}\n")

    network = ChessObscurNetwork(
        obs_planes=cfg.obs_planes,
        num_filters=cfg.num_filters,
        num_res_blocks=cfg.num_res_blocks,
        policy_head_filters=cfg.policy_head_filters,
        value_head_hidden=cfg.value_head_hidden,
        total_actions=cfg.total_actions,
    ).to(device)

    param_count = sum(p.numel() for p in network.parameters())
    print(f"[network] Parameters: {param_count:,}")

    ppo = PPOTrainer(network, cfg)
    global_step = 0

    if resume:
        if resume == "latest":
            ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
        else:
            ckpt_path = resume

        if ckpt_path and os.path.exists(ckpt_path):
            ckpt_data = load_checkpoint(ckpt_path, network, ppo.optimizer, device)
            global_step = ckpt_data.get("global_step", 0)
            print(f"[resume] Resuming from step {global_step}")
        else:
            print(f"[resume] No checkpoint found at '{resume}', starting fresh")

    if warmstart and os.path.exists(warmstart):
        warmstart_from_human_data(network, warmstart, cfg)

    # Initialize environment with curriculum starting value if enabled
    initial_max_steps = cfg.curriculum_start_steps if cfg.curriculum_enabled else cfg.max_game_steps
    env = ChessObscurEnv(cfg.num_envs, device=device, max_steps=initial_max_steps)
    obs = env.reset()

    obs_shape = (cfg.obs_planes, cfg.board_size, cfg.board_size)
    buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, obs_shape,
                           cfg.total_actions, device)

    logger = Logger(log_dir=cfg.log_dir)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    num_updates = cfg.total_timesteps // cfg.batch_size
    total_games = 0
    start_time = time.time()

    print(f"\n[train] Starting self-play PPO loop...")
    print(f"[train] {num_updates} PPO updates planned\n")

    try:
        for update in range(1, num_updates + 1):
            update_start = time.time()

            progress = global_step / cfg.total_timesteps
            ppo.update_lr(progress)

            obs, rollout_stats = collect_rollout(env, network, buffer, obs)
            rollout_data = buffer.get(next_obs=obs)

            ppo_metrics = ppo.update(rollout_data)

            global_step += cfg.batch_size
            total_games += int(rollout_stats.get("rollout/games_completed", 0))

            # Curriculum: progressively increase max_steps
            if cfg.curriculum_enabled:
                current_max_steps = cfg.curriculum_start_steps + \
                    (global_step // cfg.curriculum_every_n_timesteps) * cfg.curriculum_step_increase
                if current_max_steps != env.max_steps:
                    env.set_max_steps(current_max_steps)
                    print(f"[curriculum] Updated max_steps: {env.max_steps} → {current_max_steps} at step {global_step:,}")

            all_metrics = {**rollout_stats, **ppo_metrics}
            all_metrics["train/global_step"] = global_step
            all_metrics["train/total_games"] = total_games
            all_metrics["train/lr"] = ppo.optimizer.param_groups[0]["lr"]
            all_metrics["train/max_steps"] = env.max_steps

            update_time = time.time() - update_start
            fps = cfg.batch_size / max(update_time, 1e-6)
            all_metrics["train/fps"] = fps

            logger.log_scalars(all_metrics, global_step)

            if update % max(1, cfg.log_interval // cfg.batch_size) == 0 or update == 1:
                logger.log_console(global_step, cfg.total_timesteps, all_metrics,
                                   total_games, fps)

            if global_step % cfg.checkpoint_interval < cfg.batch_size or update == 1:
                ckpt_file = checkpoint_path(cfg.checkpoint_dir, global_step)
                save_checkpoint(ckpt_file, network, ppo.optimizer, global_step, all_metrics)

            if all_metrics.get("ppo/approx_kl", 0) > 0.05:
                print(f"[warn] High KL divergence: {all_metrics['ppo/approx_kl']:.4f}")

    except KeyboardInterrupt:
        print(f"\n[train] Interrupted at step {global_step}")

    ckpt_file = checkpoint_path(cfg.checkpoint_dir, global_step)
    save_checkpoint(ckpt_file, network, ppo.optimizer, global_step,
                    {"total_games": total_games})

    elapsed = time.time() - start_time
    print(f"\n[train] Finished. Steps: {global_step:,} | Games: {total_games:,} | "
          f"Time: {elapsed/3600:.1f}h")

    logger.close()
    return network


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Chess Obscur PPO Training")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--warmstart", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--num-res-blocks", type=int, default=None)
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--max-game-steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    cfg.device = args.device
    cfg.checkpoint_dir = args.checkpoint_dir
    if args.num_envs is not None: cfg.num_envs = args.num_envs
    if args.total_steps is not None: cfg.total_timesteps = args.total_steps
    if args.lr is not None: cfg.lr = args.lr
    if args.rollout_steps is not None: cfg.rollout_steps = args.rollout_steps
    if args.microbatch_size is not None: cfg.microbatch_size = args.microbatch_size
    if args.num_res_blocks is not None: cfg.num_res_blocks = args.num_res_blocks
    if args.num_filters is not None: cfg.num_filters = args.num_filters
    if args.eval_games is not None: cfg.eval_games = args.eval_games
    if args.max_game_steps is not None: cfg.max_game_steps = args.max_game_steps

    cfg.__post_init__()

    if "cuda" in cfg.device:
        if not torch.cuda.is_available():
            print("[error] CUDA not available! Falling back to CPU.")
            cfg.device = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[gpu] {gpu_name} | {gpu_mem:.1f} GB")

    if args.eval:
        network = ChessObscurNetwork(
            obs_planes=cfg.obs_planes,
            num_filters=cfg.num_filters,
            num_res_blocks=cfg.num_res_blocks,
            total_actions=cfg.total_actions,
        ).to(cfg.device)

        ckpt = args.checkpoint or args.resume
        if ckpt == "latest":
            ckpt = find_latest_checkpoint(cfg.checkpoint_dir)
        if ckpt and os.path.exists(ckpt):
            load_checkpoint(ckpt, network, device=cfg.device)
        else:
            print("[eval] No checkpoint specified, evaluating random network")

        evaluate(network, cfg, num_games=cfg.eval_games)
        return

    train(cfg, resume=args.resume, warmstart=args.warmstart)


if __name__ == "__main__":
    main()