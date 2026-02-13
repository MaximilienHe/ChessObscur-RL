"""
config.py — All hyperparameters for Chess Obscur PPO training.

CHANGES v4 (aggression + parry activation + draw fix):
- REWARD tuning: see reward.py for details
- entropy_coef: 0.005 -> 0.008 (CRITICAL: entropy collapsed to 1.8, need more exploration)
- entropy_coef_min: 0.0015 -> 0.003 (keep higher floor — was reaching 0.0015 and policy froze)
- entropy_coef_decay_steps: 200M -> 400M (much slower decay, entropy crashed too fast)
- clip_eps: 0.15 -> 0.18 (was too tight, clipfrac went to 0.55 during transition, now ~0.07 = under-training)
- ppo_epochs: 3 -> 4 (with wider clip, we can do more epochs again)
- num_minibatches: 8 -> 6 (slightly larger minibatches for more stable value loss)
- value_head_hidden: 256 -> 512 (value loss EXPLODED to 3.8 — needs much more capacity)
- value_coef: 1.0 -> 0.5 (reduce value loss weight to prevent value head from destabilizing policy)
- max_game_steps: 120 -> 150 (start higher, games were too short early on)
- curriculum_max_steps_cap: 300 -> 250 (cap lower — at 300, avg_length=364, too many draws)
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
    max_game_steps: int = 150     # CHANGED: 120 -> 150 initial max steps

    # ── Curriculum: progressive max_steps increase ──
    curriculum_enabled: bool = True
    curriculum_start_steps: int = 150      # CHANGED: 120 -> 150
    curriculum_step_increase: int = 10     # CHANGED: 15 -> 10 (slower increase)
    curriculum_every_n_timesteps: int = 5_000_000  # every 5M timesteps
    curriculum_max_steps_cap: int = 250    # CHANGED: 300 -> 250 (KEY FIX: 300 was causing 75% draws)

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
    value_head_hidden: int = 512     # CHANGED: 256 -> 512, value loss exploded to 3.8
    policy_head_filters: int = 32

    # ── PPO ──
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.18           # CHANGED: 0.15 -> 0.18, clipfrac was 0.07 = under-learning
    clip_value: float = 0.0          # TO DO : Might get back to 0.5 or 1.5 if too much instability
    entropy_coef: float = 0.008      # CHANGED: 0.005 -> 0.008, entropy died at 1.8
    entropy_coef_min: float = 0.003   # CHANGED: 0.0015 -> 0.003, higher floor
    entropy_coef_decay_steps: int = 400_000_000  # CHANGED: 200M -> 400M, MUCH slower
    value_coef: float = 0.25          # CHANGED: 1.0 -> 0.5, stabilize value head
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4              # CHANGED: 3 -> 4, more epochs with wider clip
    num_minibatches: int = 6         # CHANGED: 8 -> 6, larger minibatches

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