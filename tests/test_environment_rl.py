"""
Unit tests for Reinforcement Learning environment interface and 19-plane observation tensors.
"""

import pytest
import numpy as np

from chess_rl.chess_env.base import Player, Move
from chess_rl.chess_env.environments.registry import EnvironmentRegistry
from chess_rl.chess_env.environments.standard_chess import StandardChessEnvironment
from chess_rl.config.schema import EnvironmentConfig
from chess_rl.chess_env.board.types import algebraic_to_square


def test_standard_environment_lifecycle():
    env = StandardChessEnvironment()
    res = env.reset()

    assert not res.is_terminal
    assert res.reward == 0.0
    assert res.current_player == Player.WHITE
    assert len(res.legal_actions) == 20
    assert res.observation.shape == (19, 8, 8)

    # Execute move e2e4
    e2e4 = Move(from_square=12, to_square=28)
    step_res = env.step(e2e4)

    assert not step_res.is_terminal
    assert step_res.current_player == Player.BLACK
    assert len(step_res.legal_actions) == 20
    assert "e7e5" in [m.to_uci() for m in step_res.legal_actions]


def test_environment_registry():
    cfg = EnvironmentConfig(name="full_chess_8x8")
    env = EnvironmentRegistry.create(cfg)
    assert isinstance(env, StandardChessEnvironment)
    assert env.board_shape == (8, 8)
    assert env.num_channels == 19


def test_observation_tensor_feature_planes():
    env = StandardChessEnvironment()
    obs = env.get_observation_tensor()

    assert obs.shape == (19, 8, 8)
    # Check White Pawns plane (channel 0, rank 1 which is index 1)
    assert np.all(obs[0, 1, :] == 1.0)
    # Check Black Pawns plane (channel 6, rank 6 which is index 6)
    assert np.all(obs[6, 6, :] == 1.0)

    # Castling rights planes (channels 12..15) are all 1s
    assert np.all(obs[12, :, :] == 1.0)
    assert np.all(obs[13, :, :] == 1.0)
    assert np.all(obs[14, :, :] == 1.0)
    assert np.all(obs[15, :, :] == 1.0)

    # White turn plane (channel 16) is +1.0
    assert np.all(obs[16, :, :] == 1.0)


def test_terminal_rewards():
    env = StandardChessEnvironment()
    # Load Fool's Mate
    env.set_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")

    assert env.is_terminal()
    # White was mated, so White gets -1.0, Black gets +1.0
    assert env.get_reward(Player.WHITE) == -1.0
    assert env.get_reward(Player.BLACK) == 1.0
