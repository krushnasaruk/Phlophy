"""
Experiment Runner CLI Entrypoint for chess_rl.
Loads configuration, initializes tracking and hardware telemetry, and controls training lifecycle.
"""

import argparse
import sys
from pathlib import Path

# Add repository root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chess_rl.config.loader import load_experiment_config
from chess_rl.utils.seeding import seed_everything
from chess_rl.utils.logging import get_logger
from chess_rl.tracking.experiment_tracker import ExperimentTracker
from chess_rl.agents.networks.residual_net import DualHeadResNet
from chess_rl.training.curriculum.manager import CurriculumManager
from chess_rl.training.replay_buffer.uniform_buffer import UniformReplayBuffer
from chess_rl.training.optimization.trainer import Trainer
from chess_rl.training.self_play.worker import SelfPlayWorker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run self-play reinforcement learning experiment for chess_rl."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file (e.g. configs/experiment/baseline_direct.yaml)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Override experiment random seed",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Compute device ('cpu', 'cuda', or 'auto')",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Override base experiment output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize config, tracker, network, and exit without running full training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.seed is not None:
        overrides["experiment.seed"] = args.seed
    if args.device is not None:
        overrides["experiment.device"] = args.device
    if args.output_dir is not None:
        overrides["experiment.output_dir"] = args.output_dir

    # 1. Load and parse configuration
    config = load_experiment_config(args.config, overrides=overrides)
    
    # 2. Set deterministic seeds
    seed_everything(config.experiment.seed)

    # 3. Setup device
    device_str = config.experiment.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # 4. Initialize experiment tracker
    tracker = ExperimentTracker(config)
    run_dir = tracker.initialize()

    logger = tracker.logger or get_logger("run_experiment")
    logger.info(f"Loaded config: {config.experiment.name} (paradigm: {config.experiment.paradigm})")
    logger.info(f"Compute device: {device}")
    logger.info(f"Random seed: {config.experiment.seed}")

    # 5. Initialize Network Backbone
    network = DualHeadResNet.from_config(
        model_config=config.model,
        num_input_channels=config.environment.num_input_channels,
        action_space_size=config.environment.action_space_size,
        board_shape=(config.environment.board_size[0], config.environment.board_size[1]),
    ).to(device)

    logger.info(f"Constructed network: {config.model.name} with {sum(p.numel() for p in network.parameters()):,} parameters")

    # 6. Initialize Curriculum Manager
    curriculum = CurriculumManager(config.curriculum)
    if curriculum.enabled:
        logger.info(f"Curriculum enabled with {len(curriculum.stages)} stages.")
    else:
        logger.info("Direct baseline training (no curriculum).")

    # 7. Initialize Replay Buffer & Trainer
    replay_buffer = UniformReplayBuffer(capacity=config.training.replay_buffer.capacity)
    trainer = Trainer(
        network=network,
        optimizer_config=config.optimizer,
        training_config=config.training,
        device=device,
    )

    if args.dry_run:
        logger.info("Dry-run complete. Architecture, tracking, and initialization verified successfully.")
        return 0

    logger.info("Ready for self-play training loop in subsequent phase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
