"""
Baseline policy agents for evaluation and sanity verification.
"""

import random
from typing import Optional
import numpy as np

from chess_rl.chess_env.base import BaseChessEnvironment, Move
from chess_rl.agents.base import BaseAgent, AgentDecision


class RandomAgent(BaseAgent):
    """Selects uniformly random legal moves."""

    def __init__(self, name: str = "random_agent", seed: Optional[int] = None):
        self._name = name
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self._name

    def select_move(self, env: BaseChessEnvironment, temperature: float = 1.0) -> AgentDecision:
        legal_moves = env.legal_actions()
        if not legal_moves:
            raise RuntimeError("Cannot select move from terminal state with no legal actions.")
        chosen_move = self._rng.choice(legal_moves)
        
        # Build uniform probability vector over legal actions
        probs = np.zeros(env.action_space_size, dtype=np.float32)
        uniform_p = 1.0 / len(legal_moves)
        for m in legal_moves:
            probs[m.to_action_index()] = uniform_p

        return AgentDecision(
            selected_move=chosen_move,
            action_probabilities=probs,
            value_estimate=0.0,
            search_info={"policy_type": "uniform_random"},
        )
