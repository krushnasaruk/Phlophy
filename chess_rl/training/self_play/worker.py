"""
Self-play worker and episode trajectory generation.
Executes games using MCTS guided by current policy-value network checkpoints.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from chess_rl.chess_env.base import BaseChessEnvironment, Player, Move
from chess_rl.agents.networks.base import BasePolicyValueNet
from chess_rl.search.mcts.base import MCTSSearch
from chess_rl.training.replay_buffer.base import TrajectorySample
from chess_rl.config.schema import MCTSConfig


class SelfPlayWorker:
    """
    Executes self-play games and collects experience trajectories for network training.
    """

    def __init__(self, mcts_config: Optional[MCTSConfig] = None):
        self.mcts_config = mcts_config or MCTSConfig()
        self.mcts = MCTSSearch(self.mcts_config)

    def play_episode(
        self,
        env: BaseChessEnvironment,
        network: Optional[BasePolicyValueNet] = None,
        max_moves: int = 200,
    ) -> Tuple[List[TrajectorySample], Dict[str, Any]]:
        """
        Execute a single full self-play episode from start to terminal state.

        Args:
            env: Fresh or reset chess environment instance.
            network: Policy-Value network to evaluate states during MCTS.
            max_moves: Upper ply limit to prevent infinite cycles.

        Returns:
            Tuple of (list_of_trajectory_samples, episode_metadata_dict).
        """
        step_history: List[Tuple[np.ndarray, np.ndarray, Player, np.ndarray]] = []
        step_count = 0

        while not env.is_terminal() and step_count < max_moves:
            current_player = env.current_player
            obs = env.get_observation_tensor()
            legal_mask = env.get_legal_action_mask()

            # Dynamic temperature schedule: exploratory in opening plies, greedy thereafter
            temp = (
                self.mcts_config.temperature
                if step_count < self.mcts_config.temp_threshold_move
                else 0.0
            )

            # MCTS search with root Dirichlet noise for self-play exploration
            search_res = self.mcts.search(
                env=env,
                network=network,
                temperature=temp,
                add_dirichlet_noise=True,
            )

            step_history.append((obs, search_res.action_probabilities, current_player, legal_mask))
            env.step(search_res.best_move)
            step_count += 1

        # Determine terminal rewards
        trajectory: List[TrajectorySample] = []
        is_terminal = env.is_terminal()
        
        # Reward from White's perspective (+1 win, -1 loss, 0 draw)
        white_reward = env.get_reward(Player.WHITE) if is_terminal else 0.0

        for obs, pi, player, mask in step_history:
            # Reward from current step's player perspective
            player_reward = white_reward if player == Player.WHITE else -white_reward
            trajectory.append(
                TrajectorySample(
                    observation=obs,
                    action_probabilities=pi,
                    reward=float(player_reward),
                    legal_mask=mask,
                )
            )

        metadata = {
            "num_plies": step_count,
            "is_terminal": is_terminal,
            "white_reward": white_reward,
            "winner": "white" if white_reward > 0 else ("black" if white_reward < 0 else "draw"),
        }

        return trajectory, metadata
