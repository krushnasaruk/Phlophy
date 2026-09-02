"""
MCTS tree node representation and PUCT selection formula.
"""

from typing import Dict, Optional, List, Tuple
import math
import numpy as np

from chess_rl.chess_env.base import Move


class MCTSNode:
    """
    Represents a state node in the Monte Carlo Tree Search.
    Maintains edge statistics: N (visits), W (total value), Q (mean value), P (prior).
    """

    def __init__(self, parent: Optional["MCTSNode"] = None, prior: float = 1.0, action_from_parent: Optional[Move] = None):
        self.parent = parent
        self.prior = float(prior)
        self.action_from_parent = action_from_parent
        
        self.children: Dict[Move, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.mean_value: float = 0.0
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Return the mean action value Q(s, a)."""
        return self.mean_value if self.visit_count > 0 else 0.0

    def compute_puct_score(self, c_puct: float, parent_visit_count: int) -> float:
        """
        Compute Predictor Upper Confidence Bound (PUCT) score:
        Score = Q + c_puct * P * (sqrt(N_parent) / (1 + N_child))
        """
        u_score = c_puct * self.prior * (math.sqrt(parent_visit_count) / (1.0 + self.visit_count))
        return self.q_value + u_score

    def select_best_child(self, c_puct: float) -> Tuple[Move, "MCTSNode"]:
        """Select child node maximizing the PUCT score."""
        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.compute_puct_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("Cannot select best child from a node with no children.")
        return best_action, best_child

    def expand(self, action_priors: Dict[Move, float]) -> None:
        """Expand node by attaching child nodes with network prior probabilities."""
        self.is_expanded = True
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prior, action_from_parent=action)

    def backup(self, value: float) -> None:
        """
        Backpropagate evaluation value up the tree.
        Note: The value alternates perspective at each ply (-value).
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

        if self.parent is not None:
            self.parent.backup(-value)
