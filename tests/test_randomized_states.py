"""
Randomized state testing for legal move generation, action bijection, and mask consistency.
Tests thousands of generated valid legal positions without external chess datasets.
"""

import random
import pytest
import numpy as np

from chess_rl.chess_env.environments.standard_chess import StandardChessEnvironment
from chess_rl.chess_env.board.action_space import ActionEncoder, ACTION_SPACE_SIZE
from chess_rl.chess_env.rules.move_generator import MoveGenerator
from chess_rl.utils.seeding import seed_everything


def test_randomized_state_representations_and_action_masks():
    """
    Simulate random legal games to generate >2,000 diverse valid positions,
    validating legal move generation, action bijection, mask alignment, and tensor invariants.
    """
    seeds = [42, 101, 2024, 777]
    total_positions_tested = 0
    total_moves_validated = 0

    for seed in seeds:
        seed_everything(seed)
        rng = random.Random(seed)

        for game_idx in range(15):  # 15 games per seed
            env = StandardChessEnvironment()
            env.reset()

            step_count = 0
            while not env.is_terminal() and step_count < 80:
                state = env.state
                legal_moves = env.legal_actions()

                if not legal_moves:
                    break

                # 1. Action Encoding & Decoding Bijection
                encoded_indices = []
                for m in legal_moves:
                    act_id = ActionEncoder.encode(m)
                    assert 0 <= act_id < ACTION_SPACE_SIZE
                    encoded_indices.append(act_id)

                    decoded = ActionEncoder.decode(act_id)
                    assert decoded.from_square == m.from_square
                    assert decoded.to_square == m.to_square
                    if m.promotion_piece is not None:
                        assert decoded.promotion_piece == m.promotion_piece

                # 2. Collision Check: Unique action IDs for all legal moves in this position
                assert len(encoded_indices) == len(set(encoded_indices))

                # 3. Mask Alignment Check
                mask = env.get_legal_action_mask()
                assert mask.shape == (ACTION_SPACE_SIZE,)
                assert mask.sum() == len(legal_moves)
                for act_id in encoded_indices:
                    assert mask[act_id]

                # 4. Observation Tensor Invariant Check
                obs = env.get_observation_tensor()
                assert obs.shape == (19, 8, 8)
                # Max 1 piece per square
                assert np.all(obs[0:12].sum(axis=0) <= 1.0)

                # Step random move
                chosen_move = rng.choice(legal_moves)
                env.step(chosen_move)
                step_count += 1
                total_positions_tested += 1
                total_moves_validated += len(legal_moves)

    assert total_positions_tested >= 1000
    assert total_moves_validated >= 20000
