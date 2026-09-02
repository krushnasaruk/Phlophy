"""
Unit tests for experiment tracking, system telemetry, and checkpoint management.
"""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from chess_rl.config import load_experiment_config
from chess_rl.tracking import ExperimentTracker, MetricsLogger
from chess_rl.utils.system_info import collect_system_info


def test_collect_system_info():
    info = collect_system_info()
    assert "os" in info
    assert "python" in info
    assert "frameworks" in info
    assert "hardware" in info
    assert "torch" in info["frameworks"]
    assert "numpy" in info["frameworks"]


def test_experiment_tracker_initialization(tmp_path):
    cfg = load_experiment_config("configs/experiment/baseline_direct.yaml")
    tracker = ExperimentTracker(cfg, base_output_dir=tmp_path)
    run_dir = tracker.initialize()

    assert run_dir.exists()
    assert (run_dir / "checkpoints").exists()
    assert (run_dir / "plots").exists()
    assert (run_dir / "logs").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "system_info.json").exists()

    with open(run_dir / "system_info.json", "r", encoding="utf-8") as f:
        sys_info = json.load(f)
        assert sys_info["experiment"]["name"] == "baseline_direct_8x8"


def test_metrics_logger(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log({"iteration": 1, "loss_total": 2.5, "loss_value": 0.8, "loss_policy": 1.7})
    logger.log({"iteration": 2, "loss_total": 2.1, "loss_value": 0.6, "loss_policy": 1.5})

    csv_path = tmp_path / "training_metrics.csv"
    jsonl_path = tmp_path / "metrics.jsonl"

    assert csv_path.exists()
    assert jsonl_path.exists()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2


def test_checkpoint_saving_and_loading(tmp_path):
    cfg = load_experiment_config("configs/experiment/baseline_direct.yaml")
    tracker = ExperimentTracker(cfg, base_output_dir=tmp_path)
    tracker.initialize()

    dummy_model = nn.Linear(10, 2)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

    ckpt_path = tracker.save_checkpoint(
        iteration=10,
        model_state=dummy_model.state_dict(),
        optimizer_state=dummy_optimizer.state_dict(),
        extra_metadata={"elo_rating": 1250.0},
        is_best=True,
    )

    assert ckpt_path.exists()
    assert (tracker.checkpoints_dir / "model_best.pt").exists()

    loaded = tracker.load_checkpoint(ckpt_path)
    assert loaded["iteration"] == 10
    assert loaded["metadata"]["elo_rating"] == 1250.0
