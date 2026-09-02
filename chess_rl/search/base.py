"""
Abstract Base Class and data structures for search algorithms (MCTS, Minimax, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from chess_rl.chess_env.base import BaseChessEnvironment, Move


@dataclass
class SearchResult:
    """Encapsulates the output of a tree search procedure."""
    action_probabilities: np.ndarray
    best_move: Move
    root_value: float
    visit_counts: Dict[Move, int] = field(default_factory=dict)
    depth_reached: int = 0
    total_simulations: int = 0
    search_time_sec: float = 0.0


class BaseSearchAlgorithm(ABC):
    """Abstract Base Class for move search algorithms."""

    @abstractmethod
    def search(
        self,
        env: BaseChessEnvironment,
        network: Optional[Any] = None,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
    ) -> SearchResult:
        """
        Execute search on current environment state and return improved policy and value.

        Args:
            env: Current chess environment state.
            network: Policy-Value network to guide search (or None for pure rollout).
            temperature: Move visit count temperature.
            add_dirichlet_noise: Whether to add exploration noise to root priors.

        Returns:
            SearchResult with action probabilities and best move.
        """
        pass
