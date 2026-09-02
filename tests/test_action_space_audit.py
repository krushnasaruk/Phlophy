"""
Exhaustive Action Space audit, bijection proof, collision prevention, and bounds tests.
"""

import pytest
from chess_rl.chess_env.board.types import (
    PieceType,
    coords_to_square,
    square_to_algebraic,
)
from chess_rl.chess_env.board.move import Move
from chess_rl.chess_env.board.action_space import ActionEncoder, ACTION_SPACE_SIZE
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator


def test_action_space_bounds_and_rejection():
    # Valid range: 0 .. 4239
    assert ACTION_SPACE_SIZE == 4240

    with pytest.raises(ValueError):
        ActionEncoder.decode(-1)

    with pytest.raises(ValueError):
        ActionEncoder.decode(4240)

    with pytest.raises(ValueError):
        ActionEncoder.decode(5000)

    # Padding slot rejection (4228 .. 4239)
    with pytest.raises(ValueError):
        ActionEncoder.decode(4230)


def test_exhaustive_underpromotion_bijection():
    """Verify all 132 underpromotion slots (44 N + 44 B + 44 R) decode and encode perfectly."""
    promo_types = [PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK]
    seen_action_ids = set()

    for promo in promo_types:
        # White promotions (rank 7 to rank 8)
        for file in range(8):
            from_sq = coords_to_square(file, 6)
            for df in [-1, 0, 1]:
                tf = file + df
                if 0 <= tf < 8:
                    to_sq = coords_to_square(tf, 7)
                    move = Move(from_square=from_sq, to_square=to_sq, promotion_piece=promo)
                    action_id = ActionEncoder.encode(move)

                    assert 4096 <= action_id < 4228
                    assert action_id not in seen_action_ids, f"Action collision on ID {action_id} for move {move}"
                    seen_action_ids.add(action_id)

                    # Bijective decode
                    decoded_move = ActionEncoder.decode(action_id)
                    assert decoded_move.from_square == move.from_square
                    assert decoded_move.to_square == move.to_square
                    assert decoded_move.promotion_piece == move.promotion_piece

        # Black promotions (rank 2 to rank 1)
        for file in range(8):
            from_sq = coords_to_square(file, 1)
            for df in [-1, 0, 1]:
                tf = file + df
                if 0 <= tf < 8:
                    to_sq = coords_to_square(tf, 0)
                    move = Move(from_square=from_sq, to_square=to_sq, promotion_piece=promo)
                    action_id = ActionEncoder.encode(move)

                    assert 4096 <= action_id < 4228
                    assert action_id not in seen_action_ids, f"Action collision on ID {action_id} for move {move}"
                    seen_action_ids.add(action_id)

                    decoded_move = ActionEncoder.decode(action_id)
                    assert decoded_move.from_square == move.from_square
                    assert decoded_move.to_square == move.to_square
                    assert decoded_move.promotion_piece == move.promotion_piece

    assert len(seen_action_ids) == 44 * 3  # Exactly 132 unique slots


def test_no_legal_action_collisions_across_test_positions():
    """Verify that in diverse positions, every distinct legal move has a strictly unique action index."""
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "8/4P3/8/8/8/8/8/4K2k w - - 0 1",
    ]

    for fen in test_fens:
        state = GameState.from_fen(fen)
        legal_moves = MoveGenerator.generate_legal_moves(state)
        action_ids = [ActionEncoder.encode(m) for m in legal_moves]

        # Assert no duplicates (number of unique action IDs must equal number of legal moves)
        assert len(action_ids) == len(set(action_ids)), f"Collision detected in position: {fen}"
