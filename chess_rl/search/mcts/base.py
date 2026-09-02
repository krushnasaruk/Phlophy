"""
Monte Carlo Tree Search implementation guided by Policy-Value neural networks.
"""

import time
from typing import Optional, Dict, Any
import numpy as np
import torch

from chess_rl.chess_env.base import BaseChessEnvironment, Move
from chess_rl.search.base import BaseSearchAlgorithm, SearchResult
from chess_rl.search.mcts.node import MCTSNode
from chess_rl.config.schema import MCTSConfig


class MCTSSearch(BaseSearchAlgorithm):
    """
    Standard AlphaZero-style MCTS search engine.
    Executes N simulations using PUCT selection, leaf expansion with neural priors, and backpropagation.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()

    def search(
        self,
        env: BaseChessEnvironment,
        network: Optional[Any] = None,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
    ) -> SearchResult:
        start_time = time.perf_counter()
        root = MCTSNode()

        legal_moves = env.legal_actions()
        if not legal_moves:
            raise RuntimeError("Cannot execute MCTS from a state with no legal actions.")

        # Root expansion
        if network is None:
            # Uniform prior
            p = 1.0 / len(legal_moves)
            priors = {m: p for m in legal_moves}
        else:
            obs_tensor = torch.from_numpy(env.get_observation_tensor()).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(env.get_legal_action_mask()).unsqueeze(0)
            probs, value = network.predict(obs_tensor, mask_tensor)
            probs_np = probs.squeeze(0).cpu().numpy()
            
            priors = {}
            for m in legal_moves:
                idx = m.to_action_index()
                priors[m] = float(probs_np[idx])

        # Optional Dirichlet exploration noise at root
        if add_dirichlet_noise and len(legal_moves) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_moves))
            eps = self.config.dirichlet_epsilon
            for i, m in enumerate(legal_moves):
                priors[m] = (1 - eps) * priors[m] + eps * float(noise[i])

        root.expand(priors)

        # Simulation loop
        for _ in range(self.config.num_simulations):
            node = root
            sim_env = env.copy()

            # 1. Selection
            search_path = [node]
            while node.is_expanded and not sim_env.is_terminal():
                action, node = node.select_best_child(self.config.c_puct)
                sim_env.step(action)
                search_path.append(node)

            # 2. Leaf Evaluation
            if sim_env.is_terminal():
                # Terminal value from perspective of active player at leaf
                leaf_value = sim_env.get_reward(sim_env.current_player)
            else:
                leaf_moves = sim_env.legal_actions()
                if network is None:
                    leaf_value = 0.0
                    if leaf_moves:
                        p = 1.0 / len(leaf_moves)
                        leaf_priors = {m: p for m in leaf_moves}
                        node.expand(leaf_priors)
                else:
                    leaf_obs = torch.from_numpy(sim_env.get_observation_tensor()).unsqueeze(0).float()
                    leaf_mask = torch.from_numpy(sim_env.get_legal_action_mask()).unsqueeze(0)
                    leaf_probs, leaf_val_t = network.predict(leaf_obs, leaf_mask)
                    leaf_value = float(leaf_val_t.item())
                    
                    if leaf_moves:
                        probs_np = leaf_probs.squeeze(0).cpu().numpy()
                        leaf_priors = {m: float(probs_np[m.to_action_index()]) for m in leaf_moves}
                        node.expand(leaf_priors)

            # 3. Backup
            node.backup(leaf_value)

        # Build policy distribution from root visit counts
        action_probs = np.zeros(env.action_space_size, dtype=np.float32)
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            if temperature == 0:
                # Deterministic argmax
                best_action = max(visit_counts.items(), key=lambda x: x[1])[0]
                action_probs[best_action.to_action_index()] = 1.0
            else:
                # Softmax with temperature
                for action, count in visit_counts.items():
                    action_probs[action.to_action_index()] = (count ** (1.0 / temperature))
                prob_sum = action_probs.sum()
                if prob_sum > 0:
                    action_probs /= prob_sum
        else:
            best_action = legal_moves[0]
            action_probs[best_action.to_action_index()] = 1.0

        best_move = max(visit_counts.items(), key=lambda x: x[1])[0] if visit_counts else legal_moves[0]
        elapsed = time.perf_counter() - start_time

        return SearchResult(
            action_probabilities=action_probs,
            best_move=best_move,
            root_value=root.mean_value,
            visit_counts=visit_counts,
            total_simulations=self.config.num_simulations,
            search_time_sec=elapsed,
        )
