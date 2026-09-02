"""
Unit tests for configuration schema parsing, file loading, and CLI overrides.
"""

from pathlib import Path
import pytest
from chess_rl.config import load_experiment_config
from chess_rl.config.schema import ExperimentConfig


def test_load_baseline_config():
    config_path = "configs/experiment/baseline_direct.yaml"
    cfg = load_experiment_config(config_path)

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.experiment.name == "baseline_direct_8x8"
    assert cfg.experiment.paradigm == "direct"
    assert cfg.experiment.seed == 42
    assert cfg.environment.name == "full_chess_8x8"
    assert cfg.environment.board_size == [8, 8]
    assert cfg.model.name == "resnet_small"
    assert cfg.model.num_residual_blocks == 4
    assert cfg.training.mcts.num_simulations == 100
    assert cfg.optimizer.optimizer_type == "AdamW"
    assert not cfg.curriculum.enabled


def test_load_curriculum_config():
    config_path = "configs/experiment/curriculum_progressive.yaml"
    cfg = load_experiment_config(config_path)

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.experiment.name == "curriculum_progressive_staged"
    assert cfg.experiment.paradigm == "curriculum"
    assert cfg.curriculum.enabled
    assert len(cfg.curriculum.stages) == 3
    assert cfg.curriculum.stages[0].name == "endgame_kqk"
    assert cfg.curriculum.stages[1].name == "mini_chess_5x5"
    assert cfg.curriculum.stages[2].name == "full_chess_8x8"


def test_config_overrides():
    config_path = "configs/experiment/baseline_direct.yaml"
    overrides = {
        "experiment.seed": 9999,
        "model.num_residual_blocks": 6,
        "training.mcts.num_simulations": 250,
    }
    cfg = load_experiment_config(config_path, overrides=overrides)

    assert cfg.experiment.seed == 9999
    assert cfg.model.num_residual_blocks == 6
    assert cfg.training.mcts.num_simulations == 250


def test_invalid_config_path():
    with pytest.raises(FileNotFoundError):
        load_experiment_config("configs/experiment/non_existent_config.yaml")
