import torch

from config import Config
from env.chess_obscur_env import (
    ACTION_ACCEPT_LOSS,
    ChessObscurEnv,
    PHASE_DEFENSE,
    PHASE_MOVE,
)
from model.network import ChessObscurNetwork
from training.league import LeaguePool
from utils.checkpoint import prepare_model_state_dict


def test_prepare_model_state_dict_migrates_value_head():
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
    assert migration == {"action_head": False, "value_head": True}
    assert migrated_state["value_conv.0.weight"].shape == (4, 192, 1, 1)
    assert migrated_state["value_fc.0.weight"].shape == (1024, 256)
    assert torch.equal(migrated_state["value_conv.0.weight"][0], old_conv[0])
    assert torch.equal(migrated_state["value_fc.0.weight"][:, :64], old_fc)
    assert torch.equal(migrated_state["value_conv.1.weight"], torch.tensor([1.5, 1.5, 1.5, 1.5]))
    assert torch.equal(migrated_state["value_conv.1.bias"], torch.tensor([-0.25, -0.25, -0.25, -0.25]))


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
