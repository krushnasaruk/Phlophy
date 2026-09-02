"""
Unit tests for core architecture interfaces, networks, search trees, and curriculum managers.
"""

import numpy as np
import torch
import pytest

from chess_rl.config.schema import ModelConfig, CurriculumConfig, CurriculumStageConfig
from chess_rl.agents.networks.residual_net import DualHeadResNet
from chess_rl.search.mcts.node import MCTSNode
from chess_rl.chess_env.base import Move
from chess_rl.training.replay_buffer.base import TrajectorySample
from chess_rl.training.replay_buffer.uniform_buffer import UniformReplayBuffer
from chess_rl.training.curriculum.manager import CurriculumManager
from chess_rl.training.optimization.loss import AlphaZeroLoss
from chess_rl.evaluation.metrics.elo import compute_expected_score, update_elo, calculate_match_stats


def test_dual_head_resnet_forward():
    model_cfg = ModelConfig(
        num_residual_blocks=2,
        num_channels=32,
        policy_head_channels=2,
        value_head_channels=1,
        value_hidden_dim=32,
    )
    net = DualHeadResNet.from_config(
        model_config=model_cfg,
        num_input_channels=19,
        action_space_size=4096,
        board_shape=(8, 8),
    )

    batch_size = 4
    x = torch.randn(batch_size, 19, 8, 8)
    mask = torch.ones(batch_size, 4096, dtype=torch.bool)

    logits, value = net(x, mask)

    assert logits.shape == (batch_size, 4096)
    assert value.shape == (batch_size, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_mcts_node_puct_and_backup():
    root = MCTSNode()
    m1 = Move(from_square=12, to_square=28)
    m2 = Move(from_square=13, to_square=29)

    root.expand({m1: 0.7, m2: 0.3})
    assert len(root.children) == 2
    assert root.is_expanded

    # Initial selection based on prior
    best_move, best_child = root.select_best_child(c_puct=1.25)
    assert best_move == m1

    # Backup value
    best_child.backup(value=0.5)
    assert root.visit_count == 1
    assert best_child.visit_count == 1
    assert best_child.mean_value == 0.5


def test_uniform_replay_buffer():
    buffer = UniformReplayBuffer(capacity=100)
    assert len(buffer) == 0

    obs = np.zeros((19, 8, 8), dtype=np.float32)
    pi = np.ones(4096, dtype=np.float32) / 4096
    sample = TrajectorySample(observation=obs, action_probabilities=pi, reward=1.0)

    for _ in range(50):
        buffer.push(sample)

    assert len(buffer) == 50

    batch_obs, batch_pi, batch_v, _ = buffer.sample_batch(16)
    assert batch_obs.shape == (16, 19, 8, 8)
    assert batch_pi.shape == (16, 4096)
    assert batch_v.shape == (16, 1)


def test_alphazero_loss():
    criterion = AlphaZeroLoss()

    pred_logits = torch.randn(4, 100, requires_grad=True)
    pred_values = torch.randn(4, 1, requires_grad=True)
    target_pis = torch.softmax(torch.randn(4, 100), dim=-1)
    target_values = torch.tensor([[1.0], [-1.0], [0.0], [1.0]])

    total_loss, val_loss, pol_loss = criterion(pred_logits, pred_values, target_pis, target_values)

    assert total_loss.item() > 0
    assert val_loss.item() >= 0
    assert pol_loss.item() >= 0

    # Backpropagation test
    total_loss.backward()
    assert pred_logits.grad is not None
    assert pred_values.grad is not None


def test_curriculum_manager_progression():
    stage0 = CurriculumStageConfig(stage_id=0, name="stage_0", target_win_rate=0.8, min_games=10, max_games=50)
    stage1 = CurriculumStageConfig(stage_id=1, name="stage_1", target_win_rate=None, min_games=20, max_games=100)
    
    curr_cfg = CurriculumConfig(enabled=True, stages=[stage0, stage1])
    manager = CurriculumManager(curr_cfg)

    assert manager.current_stage.name == "stage_0"
    assert not manager.is_final_stage

    manager.record_games(5)
    # Below min_games -> shouldn't advance even with 100% win rate
    assert not manager.check_stage_progression(current_win_rate=1.0)

    manager.record_games(10)
    # Above min_games and met target win rate -> should advance
    advanced = manager.check_stage_progression(current_win_rate=0.85)
    assert advanced
    assert manager.current_stage.name == "stage_1"
    assert manager.is_final_stage


def test_elo_and_match_stats():
    exp_a = compute_expected_score(1500.0, 1500.0)
    assert exp_a == pytest.approx(0.5, 0.01)

    new_a, new_b = update_elo(1500.0, 1500.0, score_a=1.0, k_factor=32.0)
    assert new_a == 1516.0
    assert new_b == 1484.0

    stats = calculate_match_stats(wins=7, losses=2, draws=1)
    assert stats["total_games"] == 10
    assert stats["win_rate"] == 0.7
    assert stats["draw_rate"] == 0.1
    assert stats["score_pct"] == 0.75
