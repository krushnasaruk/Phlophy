"""
Numerically stable legal action masking utilities for policy distributions.
"""

from typing import Union
import torch
import torch.nn.functional as F


def apply_legal_action_mask(
    logits: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """
    Apply a boolean legal action mask to raw policy logits.
    Illegal actions are set to a large negative value (e.g. -1e9).

    Args:
        logits: Tensor of shape (action_space_size,) or (B, action_space_size).
        mask: Boolean tensor of same shape as logits where True indicates a legal action.
        mask_value: Replacement value for illegal actions.

    Returns:
        Masked logits tensor.
    """
    if logits.shape != mask.shape:
        raise ValueError(f"Shape mismatch: logits shape {logits.shape} != mask shape {mask.shape}")

    mask_bool = mask.bool()
    return logits.masked_fill(~mask_bool, mask_value)


def compute_masked_probabilities(
    logits: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """
    Compute numerically stable softmax probability distribution over legal actions.
    Guarantees:
      - Zero probability on illegal actions.
      - Sum of probabilities on legal actions == 1.0 (for non-terminal states).
      - Safe handling of terminal states (returns all zeros without NaNs).

    Args:
        logits: Policy logits of shape (action_space_size,) or (B, action_space_size).
        mask: Boolean legal action mask.

    Returns:
        Probability tensor of same shape as logits.
    """
    masked_logits = apply_legal_action_mask(logits, mask, mask_value=mask_value)
    
    # Check for empty masks (terminal states)
    is_batched = masked_logits.dim() > 1
    if is_batched:
        legal_counts = mask.sum(dim=-1, keepdim=True)
        # Avoid NaN for batches where a row has 0 legal moves
        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.where(legal_counts > 0, probs, torch.zeros_like(probs))
    else:
        if mask.sum() == 0:
            return torch.zeros_like(logits)
        probs = F.softmax(masked_logits, dim=-1)

    # Clean any accidental residual NaNs
    return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
