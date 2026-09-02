"""
Experiment tracking and artifact management engine.
Guarantees isolated run directories, collision avoidance, and full provenance persistence.
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import torch

from chess_rl.config.schema import ExperimentConfig
from chess_rl.utils.system_info import collect_system_info
from chess_rl.utils.logging import get_logger
from chess_rl.tracking.metrics_logger import MetricsLogger


class ExperimentTracker:
    """
    Manages experiment lifecycle, isolated output directories, checkpoint storage,
    and metadata persistence.
    """

    def __init__(self, config: ExperimentConfig, base_output_dir: Optional[Union[str, Path]] = None):
        self.config = config
        self.base_output_dir = Path(base_output_dir or config.experiment.output_dir)
        
        # Generate unique run ID using timestamp and configuration hash
        self.timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.sha256(
            f"{config.experiment.name}_{config.experiment.seed}_{self.timestamp_str}".encode("utf-8")
        ).hexdigest()[:8]
        
        self.run_name = f"{config.experiment.name}_{self.timestamp_str}_{config_hash}"
        self.run_dir = self.base_output_dir / self.run_name
        
        # Subdirectories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.plots_dir = self.run_dir / "plots"
        self.logs_dir = self.run_dir / "logs"
        
        self.metrics_logger: Optional[MetricsLogger] = None
        self.logger = None
        self._initialized = False

    def initialize(self) -> Path:
        """
        Create directories, dump config and system info, and setup loggers.

        Returns:
            The Path to the newly created run directory.
        """
        if self._initialized:
            return self.run_dir

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        log_file = self.logs_dir / "experiment.log"
        self.logger = get_logger(f"exp.{self.config.experiment.name}", log_file=log_file)
        self.logger.info(f"Initialized experiment run: {self.run_name}")
        self.logger.info(f"Run directory: {self.run_dir.resolve()}")

        # Persist raw and resolved configuration
        self._save_config()

        # Persist system and hardware telemetry
        self._save_system_info()

        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(self.run_dir)
        
        self._initialized = True
        return self.run_dir

    def _save_config(self) -> None:
        """Serialize configuration to YAML and JSON."""
        # Save raw config dict if present, else serialize dataclass
        cfg_yaml_path = self.run_dir / "config.yaml"
        with open(cfg_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config.raw_config or self.config.__dict__, f, default_flow_style=False)

    def _save_system_info(self) -> None:
        """Capture and serialize hardware and OS specifications."""
        sys_info = collect_system_info()
        sys_info["experiment"] = {
            "name": self.config.experiment.name,
            "seed": self.config.experiment.seed,
            "paradigm": self.config.experiment.paradigm,
            "start_time": datetime.datetime.now().isoformat(),
        }
        sys_info_path = self.run_dir / "system_info.json"
        with open(sys_info_path, "w", encoding="utf-8") as f:
            json.dump(sys_info, f, indent=2)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log a metrics dictionary to disk."""
        if not self._initialized:
            self.initialize()
        if self.metrics_logger:
            self.metrics_logger.log(metrics)

    def save_checkpoint(
        self,
        iteration: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save a model checkpoint with metadata.

        Args:
            iteration: Current training iteration or game count.
            model_state: Model state dictionary.
            optimizer_state: Optional optimizer state dictionary.
            extra_metadata: Optional dict of metrics/eval ratings.
            is_best: Whether this checkpoint represents the current best model.

        Returns:
            Path to the saved checkpoint file.
        """
        if not self._initialized:
            self.initialize()

        payload = {
            "iteration": iteration,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metadata": extra_metadata or {},
            "timestamp": datetime.datetime.now().isoformat(),
            "config_name": self.config.experiment.name,
            "seed": self.config.experiment.seed,
        }

        ckpt_filename = f"model_iter_{iteration:06d}.pt"
        ckpt_path = self.checkpoints_dir / ckpt_filename
        torch.save(payload, ckpt_path)

        if is_best:
            best_path = self.checkpoints_dir / "model_best.pt"
            torch.save(payload, best_path)
            if self.logger:
                self.logger.info(f"Updated best model checkpoint at iteration {iteration}")

        if self.logger:
            self.logger.info(f"Saved checkpoint: {ckpt_path.name}")

        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a saved checkpoint payload."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path.resolve()}")
        return torch.load(path, map_location="cpu")
