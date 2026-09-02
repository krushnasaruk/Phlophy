"""
Residual Convolutional Neural Network with dual Policy-Value heads.
Designed for compute-efficient tabula-rasa representation learning.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_rl.agents.networks.base import BasePolicyValueNet
from chess_rl.config.schema import ModelConfig


class ResidualBlock(nn.Module):
    """Standard 2-layer convolutional residual block with batch normalization."""

    def __init__(self, channels: int, use_batch_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class DualHeadResNet(BasePolicyValueNet):
    """
    Dual-headed ResNet architecture for policy and value predictions.
    Scalable from compact edge-sized models to deeper research configurations.
    """

    def __init__(
        self,
        num_input_channels: int = 19,
        action_space_size: int = 4096,
        board_shape: Tuple[int, int] = (8, 8),
        num_residual_blocks: int = 4,
        num_channels: int = 64,
        policy_head_channels: int = 2,
        value_head_channels: int = 1,
        value_hidden_dim: int = 64,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._action_space_size = action_space_size
        self._board_shape = board_shape

        # Initial convolutional feature extractor
        self.input_conv = nn.Conv2d(
            num_input_channels, num_channels, kernel_size=3, padding=1, bias=not use_batch_norm
        )
        self.input_bn = nn.BatchNorm2d(num_channels) if use_batch_norm else nn.Identity()

        # Residual backbone
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels, use_batch_norm=use_batch_norm) for _ in range(num_residual_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, policy_head_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_head_channels) if use_batch_norm else nn.Identity()
        policy_flat_dim = policy_head_channels * board_shape[0] * board_shape[1]
        self.policy_fc = nn.Linear(policy_flat_dim, action_space_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, value_head_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_head_channels) if use_batch_norm else nn.Identity()
        value_flat_dim = value_head_channels * board_shape[0] * board_shape[1]
        self.value_fc1 = nn.Linear(value_flat_dim, value_hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.value_fc2 = nn.Linear(value_hidden_dim, 1)

    @property
    def num_input_channels(self) -> int:
        return self._num_input_channels

    @property
    def action_space_size(self) -> int:
        return self._action_space_size

    def forward(
        self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        h = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            h = block(h)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = self.dropout(v)
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        num_input_channels: int = 19,
        action_space_size: int = 4096,
        board_shape: Tuple[int, int] = (8, 8),
    ) -> "DualHeadResNet":
        """Factory method to construct model directly from ModelConfig."""
        return cls(
            num_input_channels=num_input_channels,
            action_space_size=action_space_size,
            board_shape=board_shape,
            num_residual_blocks=model_config.num_residual_blocks,
            num_channels=model_config.num_channels,
            policy_head_channels=model_config.policy_head_channels,
            value_head_channels=model_config.value_head_channels,
            value_hidden_dim=model_config.value_hidden_dim,
            use_batch_norm=model_config.use_batch_norm,
            dropout_rate=model_config.dropout_rate,
        )
