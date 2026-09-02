"""
Experience replay buffer interfaces for off-policy self-play data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class TrajectorySample:
    """A single state-policy-value transition record."""
    observation: np.ndarray  # Shape: (C, H, W)
    action_probabilities: np.ndarray  # Shape: (action_space_size,)
    reward: float  # Value in [-1, +1] from player perspective
    legal_mask: Optional[np.ndarray] = None


class BaseReplayBuffer(ABC):
    """Abstract Base Class for experience replay buffers."""

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Maximum number of samples stored."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Current number of samples in the buffer."""
        pass

    @abstractmethod
    def push(self, sample: TrajectorySample) -> None:
        """Add a transition sample to the buffer."""
        pass

    @abstractmethod
    def push_trajectory(self, samples: List[TrajectorySample]) -> None:
        """Add a complete game episode to the buffer."""
        pass

    @abstractmethod
    def sample_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Sample a random mini-batch of transitions.

        Returns:
            Tuple of (observations, action_probs, values, legal_masks).
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored transitions."""
        pass
