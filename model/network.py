"""
network.py — Actor-Critic ResNet for Chess Obscur.
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


class ChessObscurNetwork(nn.Module):
    def __init__(self, obs_planes: int = 19, num_filters: int = 128,
                 num_res_blocks: int = 10, policy_head_filters: int = 32,
                 value_head_hidden: int = 256, total_actions: int = 4163):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(obs_planes, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, policy_head_filters, 1, bias=False),
            nn.BatchNorm2d(policy_head_filters),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(policy_head_filters * 8 * 8, total_actions)

        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(1 * 8 * 8, value_head_hidden),
            nn.ReLU(),
            nn.Linear(value_head_hidden, 1),
            nn.Tanh(),
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

    def forward(self, obs: torch.Tensor, legal_mask: torch.Tensor = None):
        x = self.input_conv(obs)
        x = self.res_blocks(x)

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
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        return self.value_fc(v).squeeze(-1)