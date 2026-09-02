"""
chess_rl: Compute-Efficient Curriculum-Based Self-Play Reinforcement Learning for Autonomous Chess.

A research-grade tabula-rasa reinforcement learning library.
"""

__version__ = "0.1.0"
__author__ = "Autonomous Chess Research Team"

from chess_rl.config.loader import load_experiment_config
from chess_rl.utils.seeding import seed_everything
from chess_rl.utils.system_info import collect_system_info
from chess_rl.tracking.experiment_tracker import ExperimentTracker

__all__ = [
    "load_experiment_config",
    "seed_everything",
    "collect_system_info",
    "ExperimentTracker",
    "__version__",
]
