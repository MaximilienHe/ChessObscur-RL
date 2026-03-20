"""
checkpoint.py — Save, load, resume training checkpoints.
"""
import os
import glob
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

LEGACY_TOTAL_ACTIONS = 4163
CURRENT_TOTAL_ACTIONS = 4099
BOARD_ACTIONS = 4096
LEGACY_DEFENSE_OFFSET = 4160


def _is_nonfinite_tensor(value: Any) -> bool:
    if not torch.is_tensor(value):
        return False
    if not (value.is_floating_point() or value.is_complex()):
        return False
    return not torch.isfinite(value).all().item()


def _find_nonfinite_model_entries(state_dict: Dict[str, torch.Tensor], limit: int = 8) -> list[str]:
    bad_entries = []
    for name, value in state_dict.items():
        if _is_nonfinite_tensor(value):
            bad_entries.append(name)
            if len(bad_entries) >= limit:
                break
    return bad_entries


def _find_nonfinite_optimizer_entries(optimizer_state_dict: Dict[str, Any],
                                      limit: int = 8) -> list[str]:
    bad_entries = []
    for param_id, slots in optimizer_state_dict.get("state", {}).items():
        for slot_name, value in slots.items():
            if _is_nonfinite_tensor(value):
                bad_entries.append(f"state[{param_id}].{slot_name}")
                if len(bad_entries) >= limit:
                    return bad_entries
    return bad_entries


def validate_model_state_dict(state_dict: Dict[str, torch.Tensor], context: str) -> None:
    bad_entries = _find_nonfinite_model_entries(state_dict)
    if bad_entries:
        raise ValueError(
            f"{context} contains non-finite tensors: {', '.join(bad_entries)}"
        )


def validate_optimizer_state_dict(optimizer_state_dict: Dict[str, Any], context: str) -> None:
    bad_entries = _find_nonfinite_optimizer_entries(optimizer_state_dict)
    if bad_entries:
        raise ValueError(
            f"{context} contains non-finite tensors: {', '.join(bad_entries)}"
        )


def save_checkpoint(path: str, network: nn.Module, optimizer: torch.optim.Optimizer,
                    global_step: int, metrics: Dict[str, Any] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Strip _orig_mod. prefix if model was compiled with torch.compile()
    state_dict = network.state_dict()
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    validate_model_state_dict(state_dict, f"checkpoint model state at step {global_step}")

    optimizer_state_dict = optimizer.state_dict()
    validate_optimizer_state_dict(
        optimizer_state_dict, f"checkpoint optimizer state at step {global_step}"
    )

    torch.save({
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "global_step": global_step,
        "metrics": metrics or {},
    }, path)
    print(f"[checkpoint] Saved: {path} (step {global_step})")


def prepare_model_state_dict(checkpoint_or_state_dict: Dict[str, Any]) -> tuple[
    Dict[str, torch.Tensor], bool, Dict[str, bool]
]:
    """Normalize compile prefixes and migrate legacy heads to the current architecture."""
    if "model_state_dict" in checkpoint_or_state_dict:
        state_dict = checkpoint_or_state_dict["model_state_dict"]
    else:
        state_dict = checkpoint_or_state_dict

    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    migration = {
        "action_head": False,
        "value_head": False,
    }

    weight_key = "policy_fc.weight"
    bias_key = "policy_fc.bias"
    if weight_key in state_dict and state_dict[weight_key].shape[0] == LEGACY_TOTAL_ACTIONS:
        migration["action_head"] = True
        weight = state_dict[weight_key]
        state_dict = dict(state_dict)
        state_dict[weight_key] = torch.cat(
            [weight[:BOARD_ACTIONS], weight[LEGACY_DEFENSE_OFFSET:]], dim=0
        )
        if bias_key in state_dict:
            bias = state_dict[bias_key]
            state_dict[bias_key] = torch.cat(
                [bias[:BOARD_ACTIONS], bias[LEGACY_DEFENSE_OFFSET:]], dim=0
            )

    # v10: migrate value head from 1 channel to 4 channels
    vconv_key = "value_conv.0.weight"
    vbn_w_key = "value_conv.1.weight"
    vbn_b_key = "value_conv.1.bias"
    vbn_rm_key = "value_conv.1.running_mean"
    vbn_rv_key = "value_conv.1.running_var"
    vfc_key = "value_fc.0.weight"
    if vconv_key in state_dict and state_dict[vconv_key].shape[0] == 1:
        migration["value_head"] = True
        state_dict = dict(state_dict)
        old_conv = state_dict[vconv_key]  # [1, C, 1, 1]
        C = old_conv.shape[1]
        new_conv = torch.zeros(4, C, 1, 1, dtype=old_conv.dtype, device=old_conv.device)
        torch.nn.init.kaiming_normal_(new_conv, mode="fan_out", nonlinearity="relu")
        new_conv[0] = old_conv[0]  # preserve first channel
        state_dict[vconv_key] = new_conv
        # Expand BatchNorm from 1→4 channels
        for bn_key in [vbn_w_key, vbn_b_key, vbn_rm_key, vbn_rv_key]:
            if bn_key in state_dict:
                old_val = state_dict[bn_key]  # [1]
                state_dict[bn_key] = old_val.repeat(4)
        # Expand value FC input from 64→256
        if vfc_key in state_dict:
            old_fc = state_dict[vfc_key]  # [hidden, 64]
            hidden = old_fc.shape[0]
            new_fc = torch.zeros(hidden, 256, dtype=old_fc.dtype, device=old_fc.device)
            torch.nn.init.xavier_uniform_(new_fc)
            new_fc[:, :64] = old_fc  # preserve weights for first channel
            state_dict[vfc_key] = new_fc
        print("[checkpoint] Migrated value head: 1 channel -> 4 channels")

    migrated = any(migration.values())
    return state_dict, migrated, migration


def load_checkpoint(path: str, network: nn.Module, optimizer: torch.optim.Optimizer = None,
                    device: str = "cuda") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict, migrated, migration = prepare_model_state_dict(ckpt)
    validate_model_state_dict(state_dict, f"checkpoint '{path}' model state")

    if optimizer is not None and "optimizer_state_dict" in ckpt and not migrated:
        validate_optimizer_state_dict(
            ckpt["optimizer_state_dict"], f"checkpoint '{path}' optimizer state"
        )

    model_keys = network.state_dict().keys()
    if any(k.startswith("_orig_mod.") for k in model_keys):
        state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}

    network.load_state_dict(state_dict)
    optimizer_state_loaded = False
    if optimizer is not None and "optimizer_state_dict" in ckpt and not migrated:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        optimizer_state_loaded = True
    elif optimizer is not None and migrated:
        print("[checkpoint] Model weights were migrated to the current architecture; optimizer state not restored")

    if migration["action_head"]:
        print("[checkpoint] Legacy checkpoint action head adapted: 4163 -> 4099")
    if migration["value_head"]:
        print("[checkpoint] Legacy checkpoint value head adapted: 1 -> 4 conv channels")

    print(f"[checkpoint] Loaded: {path} (step {ckpt.get('global_step', 0)})")
    ckpt["_model_state_migrated"] = migrated
    ckpt["_action_head_migrated"] = migration["action_head"]
    ckpt["_value_head_migrated"] = migration["value_head"]
    ckpt["_optimizer_state_loaded"] = optimizer_state_loaded
    return ckpt


def _step_from_path(path: str) -> int:
    base = os.path.basename(path)
    try:
        return int(base.replace("step_", "").replace(".pt", ""))
    except ValueError:
        return -1


def list_checkpoints(checkpoint_dir: str, descending: bool = False) -> list[str]:
    pattern = os.path.join(checkpoint_dir, "step_*.pt")
    files = glob.glob(pattern)
    files.sort(key=_step_from_path, reverse=descending)
    return files


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    files = list_checkpoints(checkpoint_dir, descending=False)
    if not files:
        return None
    return files[-1]


def checkpoint_path(checkpoint_dir: str, global_step: int) -> str:
    return os.path.join(checkpoint_dir, f"step_{global_step}.pt")
