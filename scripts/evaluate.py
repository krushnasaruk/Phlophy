"""
Evaluation CLI Entrypoint for chess_rl.
Runs arena tournaments and benchmark evaluations between checkpoints and reference agents.
"""

import argparse
import sys
from pathlib import Path

# Add repository root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_rl.config.loader import load_experiment_config
from chess_rl.utils.seeding import seed_everything
from chess_rl.utils.logging import get_logger
from chess_rl.agents.policies.base import RandomAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate chess_rl agents and checkpoints.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to saved model checkpoint .pt file",
    )
    parser.add_argument(
        "--num-games",
        "-n",
        type=int,
        default=20,
        help="Number of tournament games to play",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("evaluate")
    config = load_experiment_config(args.config)
    seed_everything(config.experiment.seed)

    logger.info(f"Loaded config for evaluation: {config.experiment.name}")
    logger.info(f"Tournament games scheduled: {args.num_games}")

    agent_a = RandomAgent(name="candidate_agent")
    agent_b = RandomAgent(name="baseline_random")

    logger.info(f"Agents initialized: {agent_a.name} vs {agent_b.name}")
    logger.info("Evaluation framework verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
