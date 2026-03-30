import torch

from config import Config
from env.chess_obscur_env import (
    ACTION_ACCEPT_LOSS,
    ChessObscurEnv,
    PHASE_DEFENSE,
    PHASE_MOVE,
)
from model.network import ChessObscurNetwork
from model.ppo import PPOTrainer
from training.league import LeaguePool
from utils.checkpoint import prepare_model_state_dict


def test_prepare_model_state_dict_migrates_value_head():
    cfg = Config(device="cpu")
    old_conv = torch.randn(1, 192, 1, 1)
    old_fc = torch.randn(1024, 64)
    state_dict = {
        "value_conv.0.weight": old_conv.clone(),
        "value_conv.1.weight": torch.tensor([1.5]),
        "value_conv.1.bias": torch.tensor([-0.25]),
        "value_conv.1.running_mean": torch.tensor([0.1]),
        "value_conv.1.running_var": torch.tensor([0.9]),
        "value_conv.1.num_batches_tracked": torch.tensor(7, dtype=torch.long),
        "value_fc.0.weight": old_fc.clone(),
    }

    migrated_state, migrated, migration = prepare_model_state_dict(state_dict)

    assert migrated is True
    assert migration["action_head"] is False
    assert migration["value_head"] is True
    assert migrated_state["value_conv.0.weight"].shape == (cfg.value_head_channels, 192, 1, 1)
    assert migrated_state["value_fc.0.weight"].shape == (1024, cfg.value_head_channels * 64)
    assert torch.equal(migrated_state["value_conv.0.weight"][0], old_conv[0])
    assert torch.equal(migrated_state["value_fc.0.weight"][:, :64], old_fc)
    assert torch.equal(migrated_state["value_conv.1.weight"][:4], torch.tensor([1.5, 1.5, 1.5, 1.5]))
    assert torch.equal(migrated_state["value_conv.1.bias"][:4], torch.tensor([-0.25, -0.25, -0.25, -0.25]))


def test_full_move_count_only_advances_during_move_phase():
    env = ChessObscurEnv(1, device="cpu", max_steps=150)

    env.phase.fill_(PHASE_DEFENSE)
    env._resolve_defense_batched = lambda actions, mask, reward: None
    env._check_endgame = lambda reward: None
    env.step(torch.tensor([ACTION_ACCEPT_LOSS]))
    assert env.full_move_count.item() == 0

    env.phase.fill_(PHASE_MOVE)
    env.full_move_count.zero_()
    env._apply_moves_batched = lambda actions, mask, reward: None
    env._check_endgame = lambda reward: None
    env.step(torch.tensor([0]))
    assert env.full_move_count.item() == 1


def test_league_pool_keeps_broad_history():
    cfg = Config(device="cpu")
    cfg.league_checkpoint_interval = 10
    cfg.league_max_checkpoints = 6
    cfg.num_filters = 32
    cfg.num_res_blocks = 1
    cfg.policy_head_filters = 8
    cfg.value_head_hidden = 32

    pool = LeaguePool(cfg)
    network = ChessObscurNetwork(
        obs_planes=cfg.obs_planes,
        num_filters=cfg.num_filters,
        num_res_blocks=cfg.num_res_blocks,
        policy_head_filters=cfg.policy_head_filters,
        value_head_hidden=cfg.value_head_hidden,
        total_actions=cfg.total_actions,
    )

    for step in range(10, 110, 10):
        pool.maybe_snapshot(network, step)

    kept_steps = [step for step, _ in pool._snapshots]

    assert len(kept_steps) == cfg.league_max_checkpoints
    assert kept_steps[-1] == 100
    assert min(kept_steps) < 50
    assert any((b - a) > cfg.league_checkpoint_interval for a, b in zip(kept_steps, kept_steps[1:]))


def test_rewards_follow_actor_perspective_not_agent_color():
    env = ChessObscurEnv(2, device="cpu", max_steps=300)

    env.board.zero_()
    env.board[:, 4] = 6
    env.board[:, 60] = 12
    env.board[:, 28] = 1
    env.board[:, 35] = 7

    env.turn_is_white[:] = True
    env.phase.fill_(PHASE_MOVE)
    env.result.zero_()
    env.castling.zero_()
    env.en_passant.fill_(-1)
    env.check_attempts.zero_()
    env.half_moves.zero_()
    env.full_move_count.zero_()
    env.pending_attacker_sq.fill_(-1)
    env.pending_target_sq.fill_(-1)
    env.parry_square.fill_(-1)
    env.pending_attacker_piece.zero_()
    env.pending_defender_piece.zero_()
    env.pending_attacker_color_white.zero_()
    env.parry_controller_is_white.zero_()
    env.agent_is_white[:] = torch.tensor([True, False])
    env.board_history[:] = env.board.unsqueeze(1).expand(-1, env.frame_stack, -1)

    obs = env._build_obs()
    assert torch.equal(obs[0], obs[1])

    capture_action = torch.tensor([28 * 64 + 35, 28 * 64 + 35])
    env.step(capture_action)
    _, reward_defense, _, _ = env.step(torch.tensor([ACTION_ACCEPT_LOSS, ACTION_ACCEPT_LOSS]))

    assert torch.allclose(reward_defense[0], reward_defense[1])


def test_gae_respects_perspective_flips():
    cfg = Config(device="cpu")
    cfg.gamma = 1.0
    cfg.gae_lambda = 1.0
    cfg.num_filters = 8
    cfg.num_res_blocks = 1
    cfg.policy_head_filters = 8
    cfg.value_head_hidden = 32
    cfg.value_head_channels = 2
    cfg.use_attention = False

    network = ChessObscurNetwork(
        obs_planes=cfg.obs_planes,
        num_filters=cfg.num_filters,
        num_res_blocks=cfg.num_res_blocks,
        policy_head_filters=cfg.policy_head_filters,
        value_head_hidden=cfg.value_head_hidden,
        total_actions=cfg.total_actions,
        value_head_channels=cfg.value_head_channels,
        use_attention=cfg.use_attention,
        attention_heads=cfg.attention_heads,
    )
    trainer = PPOTrainer(network, cfg)

    rewards = torch.tensor([[0.0], [0.0]])
    values = torch.tensor([[0.3], [0.4]])
    dones = torch.tensor([[False], [False]])
    next_value = torch.tensor([0.5])
    bootstrap_signs = torch.tensor([[-1.0], [1.0]])

    advantages, returns = trainer.compute_gae(
        rewards, values, dones, next_value, bootstrap_signs
    )

    assert torch.allclose(advantages.squeeze(1), torch.tensor([-0.8, 0.1]), atol=1e-5)
    assert torch.allclose(returns.squeeze(1), torch.tensor([-0.5, 0.5]), atol=1e-5)
