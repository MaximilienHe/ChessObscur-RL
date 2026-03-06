"""
checkpoint.py — Save, load, resume training checkpoints.
"""
import os
import glob
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


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


def load_checkpoint(path: str, network: nn.Module, optimizer: torch.optim.Optimizer = None,
                    device: str = "cuda") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle torch.compile() prefix mismatch
    state_dict = ckpt["model_state_dict"]
    model_keys = set(network.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Check if we need to strip _orig_mod. prefix
    if ckpt_keys and not model_keys & ckpt_keys:
        if any(k.startswith("_orig_mod.") for k in ckpt_keys):
            # Checkpoint was saved with torch.compile(), strip prefix
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            print(f"[checkpoint] Stripped _orig_mod. prefix from compiled checkpoint")
        elif any(k.startswith("_orig_mod.") for k in model_keys):
            # Model is compiled, add prefix to checkpoint
            state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
            print(f"[checkpoint] Added _orig_mod. prefix for compiled model")

    network.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[checkpoint] Loaded: {path} (step {ckpt.get('global_step', 0)})")
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
