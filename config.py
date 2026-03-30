"""
config.py — All hyperparameters for Chess Obscur PPO training.

CHANGES v11 (frame stacking + attention + attack planes + MCTS + tuning):
- FRAME STACKING: 4-frame history of piece planes for temporal awareness.
  obs_planes: 19 -> 58 (12 piece planes * 4 frames + 7 metadata + 3 new).
- NEW OBS PLANES: friendly/enemy x-ray attack maps + normalized move number.
- NETWORK: Spatial self-attention layer after ResNet trunk for long-range patterns.
- NETWORK: Value head channels 4 -> 8 for richer spatial representation.
- PPO: KL early stopping (threshold 0.02) replaces fixed epoch count.
- PPO: ppo_epochs 3 -> 4 (more updates with KL safety net).
- ENTROPY: Faster decay (600M steps, was 800M) — network is stronger with stacking.
- MCTS: Optional tree search at inference (num_simulations configurable).
- ROLLOUT: Obs stored in float16 to offset memory from larger obs tensor.

CHANGES v10 (training dynamics + value head):
- FIX: full_move_count only increments on PHASE_MOVE (not defense/parry).
- Value head channels 1->4 for spatial info preservation.
- Curriculum cap 180->220 for endgame learning.
- LR restart decay 0.5->0.7 to keep LR active longer.

CHANGES v8 (fresh start — audit from 493M step run):
- NETWORK: 10 blocks / 128 filters -> 15 blocks / 192 filters (~8.5M params vs ~2.5M)
- LR: linear decay to 0 -> cosine annealing with warm restarts + lr_min floor (3e-5).
- LR WARM RESTARTS: every 100M steps, LR resets to lr * 0.5 (decaying ceiling).
- LEAGUE TRAINING: 30% of games played against random past checkpoints.
- REWARD: progressive draw penalty based on game duration (see reward.py).
- REWARD: parry good_move 0.15->0.30, parry skip -0.03->-0.08 (parry under-exploited).
- value_head_hidden: 512 -> 1024 (bigger value head for bigger backbone).
- policy_head_filters: 32 -> 64 (match bigger backbone).

CHANGES v6 (training stability fix — diagnosed from TensorBoard logs):
- ROOT CAUSE FOUND: tanh on value head bounded predictions to [-1,1] but GAE returns
  reach 5-15+. This caused value_loss=16-27, corrupted advantages, KL explosion (0.23),
  and policy collapse from 85% win rate to 18% in one run.
- value head tanh REMOVED in network.py (linear output now).
- Return normalization added in ppo.py (targets normalized per-update batch).
"""
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Device ──
    device: str = "cuda"

    # ── Performance Optimization ──
    use_compile: bool = True       # torch.compile() for faster inference
    use_amp: bool = True           # Automatic Mixed Precision (float16)
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune

    # ── Environment ──
    num_envs: int = 2048         # parallel games on GPU (2048-4096 optimal on 5090)
    max_game_steps: int = 150     # initial max steps

    # ── Curriculum: progressive max_steps increase ──
    curriculum_enabled: bool = True
    curriculum_start_steps: int = 150
    curriculum_step_increase: int = 10
    curriculum_every_n_timesteps: int = 5_000_000  # every 5M timesteps
    curriculum_max_steps_cap: int = 220

    # ── Observation v11: frame stacking + attack planes + move number ──
    frame_stack: int = 4           # v11: stack last N board states for temporal awareness
    piece_planes: int = 12         # 6 friendly + 6 enemy piece types per frame
    meta_planes: int = 7           # en_passant, castling, turn, check, phase, check_attempts, half_moves
    extra_planes: int = 3          # v11: friendly_attacks, enemy_attacks, move_number
    obs_planes: int = -1           # computed: piece_planes * frame_stack + meta_planes + extra_planes
    board_size: int = 8

    # ── Action space ──
    num_board_actions: int = 4096   # 64*64 from-to
    num_defense_actions: int = 3    # block, parry, accept_loss
    total_actions: int = 4099       # 4096 + 3

    # ── Defense action indices ──
    defense_offset: int = 4096
    ACTION_ATTEMPT_BLOCK: int = 4096
    ACTION_ATTEMPT_PARRY: int = 4097
    ACTION_ACCEPT_LOSS: int = 4098

    # ── Network v11 ──
    num_res_blocks: int = 15
    num_filters: int = 192
    value_head_hidden: int = 1024
    value_head_channels: int = 8     # v11: 4 -> 8 (richer spatial info for value estimation)
    policy_head_filters: int = 64
    use_attention: bool = True       # v11: spatial self-attention after ResNet trunk
    attention_heads: int = 4         # v11: number of attention heads

    # ── PPO v11 ──
    lr: float = 3e-4
    lr_min: float = 3e-5
    lr_warmup_steps: int = 1_000_000
    lr_restart_period: int = 100_000_000
    lr_restart_decay: float = 0.7
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.12
    clip_value: float = 1.0
    entropy_coef: float = 0.015
    entropy_coef_min: float = 0.008
    entropy_coef_decay_steps: int = 600_000_000  # v11: 800M→600M, faster decay with stronger obs
    value_coef: float = 0.25
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4              # v11: 3→4 (more updates with KL early stopping safety net)
    kl_early_stop: float = 0.02      # v11: stop PPO epochs if approx_kl exceeds this
    num_minibatches: int = 8
    obs_dtype_fp16: bool = True       # v11: store obs in float16 in rollout buffer

    # ── Rollout ──
    rollout_steps: int = 256
    batch_size: int = -1
    minibatch_size: int = -1
    microbatch_size: int = 8192

    # ── Training schedule ──
    total_timesteps: int = 500_000_000
    checkpoint_interval: int = 50_000
    log_interval: int = 1_000
    eval_games: int = 100

    # ── League training v8 ──
    league_enabled: bool = True
    league_frac: float = 0.30
    league_checkpoint_interval: int = 10_000_000
    league_max_checkpoints: int = 10

    # ── MCTS v11 (inference only) ──
    mcts_enabled: bool = False
    mcts_num_simulations: int = 50
    mcts_c_puct: float = 1.5
    mcts_temperature: float = 0.0   # 0 = pick best, >0 = sample proportional to visits

    # ── Warmstart (behavioral cloning from website games) ──
    warmstart_file: Optional[str] = None
    warmstart_epochs: int = 5
    warmstart_lr: float = 1e-3
    warmstart_batch: int = 512

    # ── Paths ──
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    # ── Piece stats (from chess.js PIECE_STATS) ──
    piece_attack: dict = field(default_factory=lambda: {
        0: 1, 1: 3, 2: 3, 3: 5, 4: 9, 5: 10
    })
    piece_defense: dict = field(default_factory=lambda: {
        0: 1, 1: 3, 2: 3, 3: 7, 4: 8, 5: 10
    })
    piece_value: dict = field(default_factory=lambda: {
        0: 1, 1: 3, 2: 3, 3: 5, 4: 9, 5: 100
    })

    def __post_init__(self):
        # v11: compute obs_planes dynamically from frame_stack
        self.obs_planes = self.piece_planes * self.frame_stack + self.meta_planes + self.extra_planes
        self.batch_size = self.num_envs * self.rollout_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        if self.microbatch_size <= 0:
            self.microbatch_size = min(2048, self.minibatch_size)
        else:
            self.microbatch_size = min(self.microbatch_size, self.minibatch_size)

    def get_lr(self, global_step: int) -> float:
        """Cosine annealing with warm restarts and decaying ceiling."""
        if global_step < self.lr_warmup_steps:
            return self.lr_min + (self.lr - self.lr_min) * (global_step / self.lr_warmup_steps)

        step = global_step - self.lr_warmup_steps
        n_restarts = step // self.lr_restart_period
        step_in_cycle = step % self.lr_restart_period
        progress_in_cycle = step_in_cycle / self.lr_restart_period

        lr_max = self.lr * (self.lr_restart_decay ** n_restarts)
        lr_max = max(lr_max, self.lr_min)

        return self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * progress_in_cycle))

    def get_entropy_coef(self, global_step: int) -> float:
        """Linear decay from entropy_coef to entropy_coef_min."""
        if global_step >= self.entropy_coef_decay_steps:
            return self.entropy_coef_min
        progress = global_step / self.entropy_coef_decay_steps
        return self.entropy_coef + (self.entropy_coef_min - self.entropy_coef) * progress
