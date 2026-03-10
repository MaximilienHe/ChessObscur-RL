"""
bitpack.py — Helpers to pack boolean action masks into uint8 tensors.
"""
import torch
import torch.nn.functional as F


def packed_num_bytes(num_bits: int) -> int:
    return (num_bits + 7) // 8


def pack_action_mask(mask: torch.Tensor) -> torch.Tensor:
    """Pack a bool mask [..., A] into uint8 bytes [..., ceil(A / 8)]."""
    if mask.dtype != torch.bool:
        raise TypeError(f"Expected bool mask, got {mask.dtype}")

    original_shape = mask.shape[:-1]
    num_actions = mask.shape[-1]
    flat = mask.reshape(-1, num_actions).to(torch.uint8)

    pad = (-num_actions) % 8
    if pad:
        flat = F.pad(flat, (0, pad))

    flat = flat.view(flat.shape[0], -1, 8)
    bit_values = (1 << torch.arange(8, device=mask.device, dtype=torch.uint8)).view(1, 1, 8)
    packed = (flat * bit_values).sum(dim=-1).to(torch.uint8)
    return packed.view(*original_shape, packed.shape[-1])


def unpack_action_mask(packed: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Unpack uint8 bytes [..., B] into a bool mask [..., num_actions]."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 packed mask, got {packed.dtype}")

    original_shape = packed.shape[:-1]
    flat = packed.reshape(-1, packed.shape[-1])
    bit_shifts = torch.arange(8, device=packed.device, dtype=torch.uint8).view(1, 1, 8)
    unpacked = ((flat.unsqueeze(-1) >> bit_shifts) & 1).to(torch.bool)
    unpacked = unpacked.view(flat.shape[0], -1)[:, :num_actions]
    return unpacked.view(*original_shape, num_actions)
