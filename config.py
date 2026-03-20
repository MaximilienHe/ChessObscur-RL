"""
config.py — All hyperparameters for Chess Obscur PPO training.

CHANGES v8 (fresh start — audit from 493M step run):
- NETWORK: 10 blocks / 128 filters -> 15 blocks / 192 filters (~8.5M params vs ~2.5M)
  Bigger network = more capacity, less likely to plateau early.
- LR: linear decay to 0 -> cosine annealing with warm restarts + lr_min floor (3e-5).
  Linear decay killed training after ~200M steps (clipfrac=0, KL=0, LR≈0).
- LR WARM RESTARTS: every 100M steps, LR resets to lr * 0.5 (decaying ceiling).
  This periodically re-injects learning capacity to escape local optima.
- LEAGUE TRAINING: 30% of games played against random past checkpoints.
  Breaks the Nash draw equilibrium (79% draws) by forcing exploitation of weaker policies.
- REWARD: progressive draw penalty based on game duration (see reward.py).
- REWARD: parry good_move 0.15->0.30, parry skip -0.03->-0.08 (parry under-exploited).
- value_head_hidden: 512 -> 1024 (bigger value head for bigger backbone).
- policy_head_filters: 32 -> 64 (match bigger backbone).

CHANGES v7 (draw penalty + training throughput — resume from 122M checkpoint):
- clip_eps: 0.10 -> 0.12 (clipfrac trop bas à 0.043, le modèle sous-apprend)
- ppo_epochs: 2 -> 3 (clipfrac bas = signal qu'on peut se permettre plus de passes)
- curriculum_max_steps_cap: 250 -> 180 (parties trop longues = draw via 50-move rule)
- REWARD_WIN: 1.5 -> 2.0, REWARD_DRAW: -0.8 -> -1.3 (voir reward.py)

CHANGES v6 (training stability fix — diagnosed from TensorBoard logs):
- ROOT CAUSE FOUND: tanh on value head bounded predictions to [-1,1] but GAE returns
  reach 5-15+. This caused value_loss=16-27, corrupted advantages, KL explosion (0.23),
  and policy collapse from 85% win rate to 18% in one run.
- value head tanh REMOVED in network.py (linear output now).
- Return normalization added in ppo.py (targets normalized per-update batch).
- clip_eps: 0.18 -> 0.10 (approx_kl was exploding to 0.23, needed tighter clip)
- ppo_epochs: 4 -> 2 (fewer passes = less cumulative policy drift per rollout)
- num_minibatches: 6 -> 8 (smaller minibatches = more gradient steps but each is smaller)
- entropy_coef: 0.008 -> 0.015 (entropy died to 1.12 nats, need stronger push)
- entropy_coef_min: 0.003 -> 0.008 (higher floor, never let entropy die again)
- entropy_coef_decay_steps: 400M -> 800M (much slower decay)
- value_coef: 0.25 -> 0.25 (unchanged; with normalized targets it's already appropriate)
- REWARD: REWARD_CHECK_3RD_CAPTURE_PENALTY removed (was penalizing valid captures)
- REWARD: outcome-based check escape reward added (see reward.py / chess_obscur_env.py)
- REWARD: REWARD_PARRY_SELF_CAPTURE: -0.30 -> -0.80 (model was at 1.3% then regressed to 7%)
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
    curriculum_max_steps_cap: int = 220  # v10: 180→220, longer games for endgame learning

    # ── Observation ──
    obs_planes: int = 19
    board_size: int = 8

    # ── Action space ──
    num_board_actions: int = 4096   # 64*64 from-to
    num_defense_actions: int = 3    # block, parry, accept_loss
    total_actions: int = 4099       # 4096 + 3

    # ── Defense action indices ──
    defense_offset: int = 4096      # immediately after board actions
    ACTION_ATTEMPT_BLOCK: int = 4096
    ACTION_ATTEMPT_PARRY: int = 4097
    ACTION_ACCEPT_LOSS: int = 4098

    # ── Network v8 ──
    num_res_blocks: int = 15         # v8: 10 -> 15
    num_filters: int = 192           # v8: 128 -> 192
    value_head_hidden: int = 1024    # v8: 512 -> 1024
    policy_head_filters: int = 64    # v8: 32 -> 64

    # ── PPO ──
    lr: float = 3e-4
    lr_min: float = 3e-5            # v8: LR floor (never go to zero)
    lr_warmup_steps: int = 1_000_000  # v8: linear warmup over first 1M steps
    lr_restart_period: int = 100_000_000  # v8: cosine restart every 100M steps
    lr_restart_decay: float = 0.7    # v10: 0.5→0.7, less aggressive decay (LR stayed active longer)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.12
    clip_value: float = 1.0
    entropy_coef: float = 0.015
    entropy_coef_min: float = 0.008
    entropy_coef_decay_steps: int = 800_000_000
    value_coef: float = 0.25
    max_grad_norm: float = 0.5
    ppo_epochs: int = 3
    num_minibatches: int = 8

    # ── Rollout ──
    rollout_steps: int = 256         # steps per env before PPO update
    batch_size: int = -1             # computed = num_envs * rollout_steps
    minibatch_size: int = -1         # computed = batch_size / num_minibatches
    microbatch_size: int = 8192      # larger for RTX 5090

    # ── Training schedule ──
    total_timesteps: int = 500_000_000
    checkpoint_interval: int = 50_000
    log_interval: int = 1_000
    eval_games: int = 100

    # ── League training v8 ──
    league_enabled: bool = True
    league_frac: float = 0.30         # 30% of envs play against past checkpoints
    league_checkpoint_interval: int = 10_000_000  # snapshot every 10M steps
    league_max_checkpoints: int = 10  # keep last N snapshots in the pool

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
        0: 1, 1: 3, 2: 3, 3: 5, 4: 9, 5: 10   # p,n,b,r,q,k
    })
    piece_defense: dict = field(default_factory=lambda: {
        0: 1, 1: 3, 2: 3, 3: 7, 4: 8, 5: 10
    })
    piece_value: dict = field(default_factory=lambda: {
        0: 1, 1: 3, 2: 3, 3: 5, 4: 9, 5: 100
    })

    def __post_init__(self):
        self.batch_size = self.num_envs * self.rollout_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        if self.microbatch_size <= 0:
            self.microbatch_size = min(2048, self.minibatch_size)
        else:
            self.microbatch_size = min(self.microbatch_size, self.minibatch_size)

    def get_lr(self, global_step: int) -> float:
        """Cosine annealing with warm restarts and decaying ceiling.

        - Linear warmup for the first lr_warmup_steps.
        - After warmup: cosine annealing from lr_max to lr_min over lr_restart_period.
        - At each restart boundary, lr_max is multiplied by lr_restart_decay.
        - Never goes below lr_min.
        """
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
        """Decay linéaire de l'entropy coef de entropy_coef vers entropy_coef_min."""
        if global_step >= self.entropy_coef_decay_steps:
            return self.entropy_coef_min
        progress = global_step / self.entropy_coef_decay_steps
        return self.entropy_coef + (self.entropy_coef_min - self.entropy_coef) * progress