"""
Circular uniform experience replay buffer.
"""

from collections import deque
import random
from typing import List, Tuple, Optional
import numpy as np

from chess_rl.training.replay_buffer.base import BaseReplayBuffer, TrajectorySample


class UniformReplayBuffer(BaseReplayBuffer):
    """
    Fixed-size FIFO deque buffer with uniform random mini-batch sampling.
    """

    def __init__(self, capacity: int = 20000):
        self._capacity = capacity
        self._buffer = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, sample: TrajectorySample) -> None:
        self._buffer.append(sample)

    def push_trajectory(self, samples: List[TrajectorySample]) -> None:
        for s in samples:
            self.push(s)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if len(self._buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self._buffer)} < {batch_size}")

        batch = random.sample(self._buffer, batch_size)
        observations = np.stack([s.observation for s in batch], axis=0).astype(np.float32)
        action_probs = np.stack([s.action_probabilities for s in batch], axis=0).astype(np.float32)
        values = np.array([s.reward for s in batch], dtype=np.float32).reshape(-1, 1)

        has_masks = batch[0].legal_mask is not None
        if has_masks:
            legal_masks = np.stack([s.legal_mask for s in batch], axis=0).astype(bool)
        else:
            legal_masks = None

        return observations, action_probs, values, legal_masks

    def clear(self) -> None:
        self._buffer.clear()
