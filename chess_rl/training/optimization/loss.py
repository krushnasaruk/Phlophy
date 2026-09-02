"""
Loss functions for policy-value reinforcement learning.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_rl.config.schema import LossWeightsConfig


class AlphaZeroLoss(nn.Module):
    """
    Composite loss function for dual-headed policy-value networks:
    Loss = w_val * MSE(v_pred, v_target) + w_pol * CrossEntropy(p_logits, pi_target)
    """

    def __init__(self, weights: LossWeightsConfig = LossWeightsConfig()):
        super().__init__()
        self.weights = weights

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_values: torch.Tensor,
        target_pis: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total, value, and policy loss.

        Args:
            pred_logits: Raw policy logits of shape (B, action_space_size).
            pred_values: Tanh value predictions of shape (B, 1).
            target_pis: MCTS target policy probabilities of shape (B, action_space_size).
            target_values: Ground truth game outcomes of shape (B, 1).

        Returns:
            Tuple of (total_loss, value_loss, policy_loss).
        """
        # Value Loss: Mean Squared Error
        val_loss = F.mse_loss(pred_values, target_values)

        # Policy Loss: Cross-Entropy over probabilities
        # log_softmax is applied to raw logits
        log_probs = F.log_softmax(pred_logits, dim=-1)
        # Target cross-entropy: - sum(pi * log_prob)
        pol_loss = -torch.mean(torch.sum(target_pis * log_probs, dim=-1))

        total_loss = (
            self.weights.value_loss_weight * val_loss
            + self.weights.policy_loss_weight * pol_loss
        )

        return total_loss, val_loss, pol_loss
