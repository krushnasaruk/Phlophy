"""
Abstract base class and core contracts for chess environments.
Supports standard 8x8 chess, mini-chess variants, and endgame sub-problems.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class Player(IntEnum):
    WHITE = 1
    BLACK = -1

    @property
    def opponent(self) -> "Player":
        return Player.BLACK if self == Player.WHITE else Player.WHITE


from chess_rl.chess_env.board.move import Move


@dataclass
class StepResult:
    """Outcome of an environment step."""
    observation: np.ndarray
    reward: float
    is_terminal: bool
    current_player: Player
    legal_actions: List[Move]
    info: Dict[str, Any] = field(default_factory=dict)


class BaseChessEnvironment(ABC):
    """
    Abstract Base Class for all chess environment variations.
    Implements standard Gym-like contracts with full legal move masking.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Environment name identifier."""
        pass

    @property
    @abstractmethod
    def board_shape(self) -> Tuple[int, int]:
        """(height, width) of the active board."""
        pass

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Number of feature planes in the observation tensor."""
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Total size of discrete action space."""
        pass

    @property
    @abstractmethod
    def current_player(self) -> Player:
        """Active player whose turn it is to move."""
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> StepResult:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action: Move) -> StepResult:
        """Execute a move and return the transition result."""
        pass

    @abstractmethod
    def legal_actions(self) -> List[Move]:
        """Return list of all strictly legal moves from current state."""
        pass

    @abstractmethod
    def get_legal_action_mask(self) -> np.ndarray:
        """Return a binary 1D numpy array of shape (action_space_size,) where 1=legal, 0=illegal."""
        pass

    @abstractmethod
    def get_observation_tensor(self) -> np.ndarray:
        """Return float32 tensor of shape (C, H, W) representing the board state."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if game has ended (checkmate, stalemate, draw)."""
        pass

    @abstractmethod
    def get_reward(self, player: Player) -> float:
        """Return terminal reward from perspective of specified player (+1, -1, 0)."""
        pass

    @abstractmethod
    def copy(self) -> "BaseChessEnvironment":
        """Return a deep copy of the current environment state."""
        pass

    @abstractmethod
    def render(self, mode: str = "text") -> Optional[str]:
        """Render board representation as text or surface."""
        pass
