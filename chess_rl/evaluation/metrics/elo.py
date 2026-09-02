"""
Elo rating and statistical comparison utilities.
Calculates relative playing strength without external engine reliance.
"""

import math
from typing import Tuple, Dict, Any


def compute_expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float = 32.0,
) -> Tuple[float, float]:
    """
    Update Elo ratings for two players given the match outcome score for player A (1.0 win, 0.5 draw, 0.0 loss).

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    expected_a = compute_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * (score_b - expected_b)

    return new_rating_a, new_rating_b


def calculate_match_stats(wins: int, losses: int, draws: int) -> Dict[str, Any]:
    """Calculate win rate, score percentage, and error bounds."""
    total = wins + losses + draws
    if total == 0:
        return {"total_games": 0, "win_rate": 0.0, "draw_rate": 0.0, "score_pct": 0.0}

    score = wins + 0.5 * draws
    score_pct = score / total
    win_rate = wins / total
    draw_rate = draws / total
    loss_rate = losses / total

    # Standard error for proportion
    se = math.sqrt(score_pct * (1.0 - score_pct) / total) if total > 1 else 0.0

    return {
        "total_games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "score_pct": score_pct,
        "standard_error": se,
        "ci_95_lower": max(0.0, score_pct - 1.96 * se),
        "ci_95_upper": min(1.0, score_pct + 1.96 * se),
    }
