"""
Abstract agent interfaces and decision contracts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from chess_rl.chess_env.base import BaseChessEnvironment, Move


@dataclass
class AgentDecision:
    """Encapsulates the decision made by an agent for a given position."""
    selected_move: Move
    action_probabilities: Optional[np.ndarray] = None
    value_estimate: Optional[float] = None
    search_info: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract Base Class for chess agents (Random, MCTS-Guided, Neural, Benchmark)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name or version tag of the agent."""
        pass

    @abstractmethod
    def select_move(self, env: BaseChessEnvironment, temperature: float = 1.0) -> AgentDecision:
        """
        Select a move given the current environment state.

        Args:
            env: Active chess environment.
            temperature: Exploration temperature for action selection.

        Returns:
            AgentDecision containing selected Move and metadata.
        """
        pass

    def reset(self) -> None:
        """Reset internal agent memory or search caches."""
        pass
