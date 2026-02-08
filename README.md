# Chess Obscur — PPO RL Training Pipeline (Full GPU)

## Architecture

```
chess_obscur_rl/
├── env/
│   ├── chess_obscur_env.py      # Vectorized GPU environment (PyTorch tensors on CUDA)
│   ├── move_tables.py           # Precomputed move lookup tables (GPU)
│   └── reward.py                # Reward shaping
├── model/
│   ├── network.py               # Actor-Critic ResNet
│   └── ppo.py                   # PPO algorithm (clipped, GAE, entropy bonus)
├── training/
│   ├── train.py                 # Main continuous training loop
│   ├── self_play.py             # Self-play data generation (both sides)
│   └── import_ndjson.py         # Import website games for behavioral cloning warmstart
├── utils/
│   ├── checkpoint.py            # Save/load/resume checkpoints
│   └── logger.py                # TensorBoard + console metrics
├── config.py                    # All hyperparameters
├── requirements.txt
└── README.md
```

## Key Design Decisions

### Full GPU Pipeline
- Board state = `(N, 19, 8, 8)` float32 tensor on CUDA — never leaves GPU during rollout
- Legal move mask computed on GPU via precomputed attack/move tables
- No Python loops over individual environments — everything is batched

### Chess Obscur Specifics Handled
- **QTE Defense**: Modeled as a discrete choice among ATTEMPT_BLOCK, ATTEMPT_PARRY, ACCEPT_LOSS
- **Parry Move**: When in parry_move phase, action space switches to controlling attacker's piece
- **3-Check Rule**: State tracks check attempts, agent must learn to escape check efficiently
- **Stochastic Captures**: tau = DEF/(ATK+DEF) probability modeled explicitly — agent learns risk/reward

### Action Space (4096 + 64 + 3 = 4163 actions)
- 64 source × 64 target = 4096 (move/capture/parry)
- 64 underpromotions (knight=0-15, bishop=16-31, rook=32-47, spare=48-63)
- 3 defense actions: ATTEMPT_BLOCK, ATTEMPT_PARRY, ACCEPT_LOSS
- action index: move = from*64+to, defense = 4096+60+{0,1,2}

### Observation Space: 19 planes of 8x8
| Planes | Content |
|--------|---------|
| 0-5    | Own pieces (P,N,B,R,Q,K) binary |
| 6-11   | Opponent pieces |
| 12     | En passant square |
| 13     | Castling rights (4 bits broadcast) |
| 14     | Turn indicator (all 1s = white to move) |
| 15     | Check indicator |
| 16     | Phase (0=move, 0.5=defense, 1=parry_move) |
| 17     | Check attempts counter (normalized /3) |
| 18     | Half-move clock (normalized /100) |

## Usage

```bash
# Install (WSL with CUDA 12+)
pip install -r requirements.txt

# Continuous self-play training
python -m training.train --device cuda --num-envs 512 --resume latest

# Import website games for warmstart
python -m training.import_ndjson --file dataset.ndjson --output data/human_games.pt

# Warmstart then continue
python -m training.train --device cuda --warmstart data/human_games.pt

# Evaluate
python -m training.train --eval --checkpoint checkpoints/step_100000.pt
```

## Hardware Notes
- 5080/5090: use `--num-envs 1024` or `2048` for max GPU occupancy
- All tensors stay on CUDA, zero CPU-GPU copies during rollout
- Env step is pure tensor ops, ~50k+ steps/sec on 5090 with 1024 envs
