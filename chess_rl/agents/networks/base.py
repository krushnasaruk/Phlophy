"""
Abstract base class for Policy-Value Neural Networks.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn


class BasePolicyValueNet(nn.Module, ABC):
    """
    Abstract Base Class for dual-headed Policy-Value Neural Networks.
    Takes board observation tensors of shape (B, C, H, W) and outputs:
      - Policy logits: shape (B, action_space_size)
      - Value estimate: shape (B, 1) in [-1, +1]
    """

    @property
    @abstractmethod
    def num_input_channels(self) -> int:
        """Expected number of input channels."""
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Total size of action space."""
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Batch tensor of observations of shape (B, C, H, W).
            legal_mask: Optional boolean or binary mask of shape (B, action_space_size).

        Returns:
            Tuple of (policy_logits, value) tensors.
        """
        pass

    def predict(
        self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference forward pass with evaluation mode and softmax probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits, value = self.forward(x, legal_mask=legal_mask)
            if legal_mask is not None:
                # Mask illegal actions with large negative value
                logits = logits.masked_fill(~legal_mask.bool(), -1e9)
            probs = torch.softmax(logits, dim=-1)
        return probs, value
