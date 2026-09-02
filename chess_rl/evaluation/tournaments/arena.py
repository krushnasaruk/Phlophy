"""
Tournament Arena for head-to-head agent evaluation.
Alternates colors across games to eliminate first-mover advantage bias.
"""

from typing import Dict, Any, Tuple
from chess_rl.chess_env.base import BaseChessEnvironment, Player
from chess_rl.agents.base import BaseAgent
from chess_rl.evaluation.metrics.elo import calculate_match_stats
from chess_rl.utils.logging import get_logger


class Arena:
    """
    Evaluates two agents by playing a symmetric head-to-head tournament.
    """

    def __init__(self, agent_a: BaseAgent, agent_b: BaseAgent, env: BaseChessEnvironment):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.env = env
        self.logger = get_logger("arena")

    def play_game(self, agent_white: BaseAgent, agent_black: BaseAgent, max_moves: int = 200) -> Tuple[float, int]:
        """
        Play a single game between two agents.

        Returns:
            Tuple of (reward_for_white, move_count).
        """
        self.env.reset()
        step_count = 0

        while not self.env.is_terminal() and step_count < max_moves:
            current_agent = agent_white if self.env.current_player == Player.WHITE else agent_black
            decision = current_agent.select_move(self.env, temperature=0.0)
            self.env.step(decision.selected_move)
            step_count += 1

        reward_white = self.env.get_reward(Player.WHITE) if self.env.is_terminal() else 0.0
        return reward_white, step_count

    def play_match(self, num_games: int = 20, max_moves: int = 200) -> Dict[str, Any]:
        """
        Play a full match of num_games alternating colors.

        Returns:
            Dictionary with win, loss, draw counts and statistical metrics for agent_a.
        """
        wins_a = 0
        losses_a = 0
        draws = 0

        for game_idx in range(num_games):
            # Alternate colors: even -> A is White, odd -> A is Black
            if game_idx % 2 == 0:
                reward_white, _ = self.play_game(self.agent_a, self.agent_b, max_moves=max_moves)
                if reward_white > 0:
                    wins_a += 1
                elif reward_white < 0:
                    losses_a += 1
                else:
                    draws += 1
            else:
                reward_white, _ = self.play_game(self.agent_b, self.agent_a, max_moves=max_moves)
                if reward_white > 0:
                    losses_a += 1
                elif reward_white < 0:
                    wins_a += 1
                else:
                    draws += 1

        stats = calculate_match_stats(wins_a, losses_a, draws)
        stats["agent_a_name"] = self.agent_a.name
        stats["agent_b_name"] = self.agent_b.name
        return stats
