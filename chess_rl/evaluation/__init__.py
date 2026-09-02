"""
Evaluation, tournament, and benchmark metrics subpackage.
"""

from chess_rl.evaluation.metrics.elo import compute_expected_score, update_elo, calculate_match_stats
from chess_rl.evaluation.tournaments.arena import Arena
from chess_rl.evaluation.benchmarks.tactical_suite import TacticalBenchmarkSuite

__all__ = [
    "compute_expected_score",
    "update_elo",
    "calculate_match_stats",
    "Arena",
    "TacticalBenchmarkSuite",
]
