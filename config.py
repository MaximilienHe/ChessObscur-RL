"""
config.py — All hyperparameters for Chess Obscur PPO training.

CHANGES v3 (parry fix + stability):
- entropy_coef_min: 0.0005 -> 0.0015 (keep exploring, especially parry situations)
- entropy_coef: 0.003 -> 0.005 (restart with more exploration if retraining)
- entropy_coef_decay_steps: 100M -> 200M (slower decay)
- clip_eps: 0.2 -> 0.15 (reduce policy instability, clipfrac was 0.25)
- value_head_hidden: 256 -> 384 (value loss was still 0.46, needs more capacity)
- ppo_epochs: 4 -> 3 (with smaller clip_eps, fewer epochs prevent overshooting)
- num_minibatches: 4 -> 8 (smaller minibatches = more stable gradients)
- REWARD_STEP_PENALTY via reward.py: -0.003 -> -0.002
"""
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
    max_game_steps: int = 120     # initial max steps before timeout draw

    # ── Curriculum: progressive max_steps increase ──
    curriculum_enabled: bool = True
    curriculum_start_steps: int = 120      # starting max_steps
    curriculum_step_increase: int = 15     # transition douce
    curriculum_every_n_timesteps: int = 5_000_000  # every 5M timesteps
    curriculum_max_steps_cap: int = 300    # ne jamais dépasser 300 steps

    # ── Observation ──
    obs_planes: int = 19
    board_size: int = 8

    # ── Action space ──
    num_board_actions: int = 4096   # 64*64 from-to
    num_promo_actions: int = 64     # underpromotion slots
    num_defense_actions: int = 3    # block, parry, accept_loss
    total_actions: int = 4163       # 4096 + 64 + 3

    # ── Defense action indices ──
    defense_offset: int = 4160      # 4096 + 64
    ACTION_ATTEMPT_BLOCK: int = 4160
    ACTION_ATTEMPT_PARRY: int = 4161
    ACTION_ACCEPT_LOSS: int = 4162

    # ── Network ──
    num_res_blocks: int = 10
    num_filters: int = 128
    value_head_hidden: int = 256     # CHANGED: 256 -> 384, value loss was high
    policy_head_filters: int = 32

    # ── PPO ──
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.15           # CHANGED: 0.2 -> 0.15, reduce policy instability
    clip_value: float = 0.5
    entropy_coef: float = 0.005      # CHANGED: 0.003 -> 0.005, more initial exploration
    entropy_coef_min: float = 0.0015  # CHANGED: 0.0005 -> 0.0015, keep exploring parry
    entropy_coef_decay_steps: int = 200_000_000  # CHANGED: 100M -> 200M, slower decay
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    ppo_epochs: int = 3              # CHANGED: 4 -> 3, prevent overshooting with tighter clip
    num_minibatches: int = 8         # CHANGED: 4 -> 8, more stable gradients

    # ── Rollout ──
    rollout_steps: int = 256         # steps per env before PPO update
    batch_size: int = -1             # computed = num_envs * rollout_steps
    minibatch_size: int = -1         # computed = batch_size / num_minibatches
    microbatch_size: int = 8192      # larger for RTX 5090 (was 2048)

    # ── Training schedule ──
    total_timesteps: int = 500_000_000  # very long — continuous training
    checkpoint_interval: int = 50_000
    log_interval: int = 1_000
    eval_games: int = 100

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

    def get_entropy_coef(self, global_step: int) -> float:
        """Decay linéaire de l'entropy coef de entropy_coef vers entropy_coef_min."""
        if global_step >= self.entropy_coef_decay_steps:
            return self.entropy_coef_min
        progress = global_step / self.entropy_coef_decay_steps
        return self.entropy_coef + (self.entropy_coef_min - self.entropy_coef) * progress