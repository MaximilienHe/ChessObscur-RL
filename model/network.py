"""
network.py — Actor-Critic ResNet with spatial attention for Chess Obscur.

CHANGES v11:
- SpatialAttention: multi-head self-attention over 8x8 board positions after ResNet trunk.
  Captures long-range piece interactions (e.g. bishop pin across the board) that
  local 3x3 convolutions struggle with.
- Value head channels: configurable (default 8, was 4) for richer spatial representation.
- obs_planes now dynamic (58 with frame_stack=4 vs 19 previously).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class SpatialAttention(nn.Module):
    """Multi-head self-attention over spatial positions (8x8 = 64 tokens).

    Each spatial position is treated as a token with `channels` features.
    This allows the network to model long-range dependencies between distant
    squares (e.g. a rook on a1 controlling h1) without stacking many conv layers.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to (B, H*W, C) — each spatial position becomes a token
        tokens = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 64, C)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        out = self.norm(tokens + attn_out)  # residual + LayerNorm
        return out.permute(0, 2, 1).contiguous().view(B, C, H, W)


class ChessObscurNetwork(nn.Module):
    def __init__(self, obs_planes: int = 58, num_filters: int = 192,
                 num_res_blocks: int = 15, policy_head_filters: int = 64,
                 value_head_hidden: int = 1024, total_actions: int = 4099,
                 value_head_channels: int = 8, use_attention: bool = True,
                 attention_heads: int = 4):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(obs_planes, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # v11: optional spatial self-attention after ResNet trunk
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention(num_filters, attention_heads)

        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, policy_head_filters, 1, bias=False),
            nn.BatchNorm2d(policy_head_filters),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(policy_head_filters * 8 * 8, total_actions)

        # v11: value head channels configurable (default 8, was 4 in v10)
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, value_head_channels, 1, bias=False),
            nn.BatchNorm2d(value_head_channels),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(value_head_channels * 8 * 8, value_head_hidden),
            nn.ReLU(),
            nn.Linear(value_head_hidden, 1),
            # No Tanh: returns can exceed [-1,1] with intermediate rewards accumulated
            # over long games. Tanh caused value_loss=16-27 and KL explosion.
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor, legal_mask: torch.Tensor = None):
        x = self.input_conv(obs)
        x = self.res_blocks(x)

        # v11: spatial attention for long-range dependencies
        if self.use_attention:
            x = self.attention(x)

        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        if legal_mask is not None:
            # Use -1e4 instead of -1e8 for float16 compatibility (AMP)
            policy_logits = policy_logits.masked_fill(~legal_mask, -1e4)

        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        value = self.value_fc(v)

        return policy_logits, value

    def get_action_and_value(self, obs: torch.Tensor, legal_mask: torch.Tensor,
                              action: torch.Tensor = None):
        policy_logits, value = self.forward(obs, legal_mask)
        dist = torch.distributions.Categorical(logits=policy_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(obs)
        x = self.res_blocks(x)
        if self.use_attention:
            x = self.attention(x)
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        return self.value_fc(v).squeeze(-1)
