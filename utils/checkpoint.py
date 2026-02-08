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
    torch.save({
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "metrics": metrics or {},
    }, path)
    print(f"[checkpoint] Saved: {path} (step {global_step})")


def load_checkpoint(path: str, network: nn.Module, optimizer: torch.optim.Optimizer = None,
                    device: str = "cuda") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    network.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[checkpoint] Loaded: {path} (step {ckpt.get('global_step', 0)})")
    return ckpt


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    pattern = os.path.join(checkpoint_dir, "step_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    def step_from_path(p):
        base = os.path.basename(p)
        try:
            return int(base.replace("step_", "").replace(".pt", ""))
        except ValueError:
            return 0
    files.sort(key=step_from_path)
    return files[-1]


def checkpoint_path(checkpoint_dir: str, global_step: int) -> str:
    return os.path.join(checkpoint_dir, f"step_{global_step}.pt")