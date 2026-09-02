"""
Neural network architectures and model definitions for chess_rl.
"""

from chess_rl.agents.networks.base import BasePolicyValueNet
from chess_rl.agents.networks.residual_net import ResidualBlock, DualHeadResNet

__all__ = [
    "BasePolicyValueNet",
    "ResidualBlock",
    "DualHeadResNet",
]
