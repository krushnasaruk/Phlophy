"""
Agent, network, and policy interfaces for chess_rl.
"""

from chess_rl.agents.base import BaseAgent, AgentDecision
from chess_rl.agents.networks.base import BasePolicyValueNet
from chess_rl.agents.networks.residual_net import ResidualBlock, DualHeadResNet
from chess_rl.agents.policies.base import RandomAgent
from chess_rl.agents.policies.random_masked_agent import RandomMaskedAgent
from chess_rl.agents.policies.masking import apply_legal_action_mask, compute_masked_probabilities

__all__ = [
    "BaseAgent",
    "AgentDecision",
    "BasePolicyValueNet",
    "ResidualBlock",
    "DualHeadResNet",
    "RandomAgent",
    "RandomMaskedAgent",
    "apply_legal_action_mask",
    "compute_masked_probabilities",
]
