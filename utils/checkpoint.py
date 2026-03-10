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


def save_checkpoint(path: str, network: nn.Module, optimizer: torch.optim.Optimizer,
                    global_step: int, metrics: Dict[str, Any] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Strip _orig_mod. prefix if model was compiled with torch.compile()
    state_dict = network.state_dict()
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    torch.save({
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "metrics": metrics or {},
    }, path)
    print(f"[checkpoint] Saved: {path} (step {global_step})")


def prepare_model_state_dict(checkpoint_or_state_dict: Dict[str, Any]) -> tuple[Dict[str, torch.Tensor], bool]:
    """Normalize compile prefixes and migrate the legacy 4163-action head to 4099."""
    if "model_state_dict" in checkpoint_or_state_dict:
        state_dict = checkpoint_or_state_dict["model_state_dict"]
    else:
        state_dict = checkpoint_or_state_dict

    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    migrated = False

    weight_key = "policy_fc.weight"
    bias_key = "policy_fc.bias"
    if weight_key in state_dict and state_dict[weight_key].shape[0] == LEGACY_TOTAL_ACTIONS:
        migrated = True
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

    return state_dict, migrated


def load_checkpoint(path: str, network: nn.Module, optimizer: torch.optim.Optimizer = None,
                    device: str = "cuda") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict, migrated = prepare_model_state_dict(ckpt)

    model_keys = network.state_dict().keys()
    if any(k.startswith("_orig_mod.") for k in model_keys):
        state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}

    network.load_state_dict(state_dict)
    optimizer_state_loaded = False
    if optimizer is not None and "optimizer_state_dict" in ckpt and not migrated:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        optimizer_state_loaded = True
    elif optimizer is not None and migrated:
        print("[checkpoint] Migrated legacy 4163-action policy head to 4099; optimizer state not restored")

    if migrated:
        print("[checkpoint] Legacy checkpoint action head adapted: 4163 -> 4099")

    print(f"[checkpoint] Loaded: {path} (step {ckpt.get('global_step', 0)})")
    ckpt["_action_head_migrated"] = migrated
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
