"""
Random Masked Agent for representation and environment validation.
Selects uniformly at random strictly from the boolean legal action mask.
"""

import random
from typing import Optional, List
import numpy as np

from chess_rl.chess_env.base import BaseChessEnvironment, Move
from chess_rl.chess_env.board.action_space import ActionEncoder
from chess_rl.agents.base import BaseAgent, AgentDecision


class RandomMaskedAgent(BaseAgent):
    """
    Validation agent that samples strictly from the environment's legal action mask.
    Guaranteed to produce zero illegal actions.
    """

    def __init__(self, name: str = "random_masked_agent", seed: Optional[int] = None):
        self._name = name
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self._name

    def select_move(self, env: BaseChessEnvironment, temperature: float = 1.0) -> AgentDecision:
        """
        Select move by sampling an active action index from get_legal_action_mask().
        """
        legal_mask = env.get_legal_action_mask()
        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            raise RuntimeError("Cannot select move from terminal state with no legal actions in mask.")

        # Uniform random selection from legal indices
        chosen_idx = int(self._rng.choice(legal_indices))

        # Build probability vector
        probs = np.zeros(env.action_space_size, dtype=np.float32)
        probs[legal_indices] = 1.0 / len(legal_indices)

        # Retrieve exact legal move matching chosen_idx
        env_legal_moves = env.legal_actions()
        matching_move = None
        for m in env_legal_moves:
            if ActionEncoder.encode(m) == chosen_idx:
                matching_move = m
                break

        if matching_move is None:
            # Fallback to decode
            state_obj = getattr(env, "state", None)
            matching_move = ActionEncoder.decode(chosen_idx, state=state_obj)

        return AgentDecision(
            selected_move=matching_move,
            action_probabilities=probs,
            value_estimate=0.0,
            search_info={
                "action_id": chosen_idx,
                "num_legal_actions": len(legal_indices),
                "policy_type": "masked_uniform_random",
            },
        )
