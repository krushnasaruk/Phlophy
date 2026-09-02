"""
Unit tests for neural network observation tensor ingestion, policy masking, and numerical stability.
"""

import pytest
import torch
import numpy as np

from chess_rl.agents.networks.residual_net import DualHeadResNet
from chess_rl.agents.policies.masking import apply_legal_action_mask, compute_masked_probabilities
from chess_rl.chess_env.board.action_space import ACTION_SPACE_SIZE
from chess_rl.chess_env.environments.standard_chess import StandardChessEnvironment


def test_neural_network_input_output_contracts():
    net = DualHeadResNet(
        num_input_channels=19,
        action_space_size=ACTION_SPACE_SIZE,
        board_shape=(8, 8),
        num_residual_blocks=2,
        num_channels=32,
    )
    net.eval()

    # Single observation
    x_single = torch.randn(1, 19, 8, 8)
    logits_s, val_s = net(x_single)
    assert logits_s.shape == (1, ACTION_SPACE_SIZE)
    assert val_s.shape == (1, 1)
    assert -1.0 <= val_s.item() <= 1.0

    # Batched observation
    batch_size = 8
    x_batch = torch.randn(batch_size, 19, 8, 8)
    mask_batch = torch.ones(batch_size, ACTION_SPACE_SIZE, dtype=torch.bool)
    logits_b, val_b = net(x_batch, mask_batch)
    assert logits_b.shape == (batch_size, ACTION_SPACE_SIZE)
    assert val_b.shape == (batch_size, 1)


def test_policy_masking_numerical_stability():
    logits = torch.tensor([2.0, -1.0, 0.5, 3.0, 1.2])
    mask = torch.tensor([True, False, True, False, False], dtype=torch.bool)

    masked_logits = apply_legal_action_mask(logits, mask, mask_value=-1e9)
    assert masked_logits[0].item() == 2.0
    assert masked_logits[2].item() == 0.5
    assert masked_logits[1].item() == -1e9
    assert masked_logits[3].item() == -1e9
    assert masked_logits[4].item() == -1e9

    probs = compute_masked_probabilities(logits, mask)
    assert probs[1].item() == 0.0
    assert probs[3].item() == 0.0
    assert probs[4].item() == 0.0
    assert probs[0].item() > 0.0
    assert probs[2].item() > 0.0
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    assert not torch.isnan(probs).any()


def test_terminal_empty_mask_handling():
    # Empty mask with all False (terminal state)
    logits = torch.randn(ACTION_SPACE_SIZE)
    empty_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)

    probs = compute_masked_probabilities(logits, empty_mask)
    assert not torch.isnan(probs).any()
    assert torch.all(probs == 0.0)

    # Batched with one empty row
    b_logits = torch.randn(3, ACTION_SPACE_SIZE)
    b_mask = torch.ones(3, ACTION_SPACE_SIZE, dtype=torch.bool)
    b_mask[1] = False  # Row 1 is empty

    b_probs = compute_masked_probabilities(b_logits, b_mask)
    assert not torch.isnan(b_probs).any()
    assert torch.isclose(b_probs[0].sum(), torch.tensor(1.0))
    assert torch.all(b_probs[1] == 0.0)
    assert torch.isclose(b_probs[2].sum(), torch.tensor(1.0))
