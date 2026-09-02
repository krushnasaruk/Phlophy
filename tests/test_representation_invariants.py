"""
Unit tests for observation tensor representation invariants.
Validates piece conservation, spatial non-overlap, color polarity, castling, and en passant accuracy.
"""

import pytest
import numpy as np

from chess_rl.chess_env.board.types import Color, PieceType, Piece, algebraic_to_square, square_to_coords
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator
from chess_rl.chess_env.representation.tensor_encoder import StandardTensorEncoder


def test_piece_conservation_invariant():
    encoder = StandardTensorEncoder()
    state = GameState.initial()
    tensor = encoder.encode(state)

    # Initial position: exactly 16 white pieces and 16 black pieces
    white_piece_bits = tensor[0:6].sum()
    black_piece_bits = tensor[6:12].sum()
    total_piece_bits = tensor[0:12].sum()

    assert white_piece_bits == 16.0
    assert black_piece_bits == 16.0
    assert total_piece_bits == 32.0

    # Exactly one white king (plane 5) and one black king (plane 11)
    assert tensor[5].sum() == 1.0
    assert tensor[11].sum() == 1.0


def test_empty_squares_and_non_overlap_invariant():
    encoder = StandardTensorEncoder()
    state = GameState.initial()
    tensor = encoder.encode(state)

    # Sum of all piece planes across channels 0..11 at any square must be <= 1.0
    square_occupancy = tensor[0:12].sum(axis=0)
    assert np.all(square_occupancy <= 1.0)
    assert np.all((square_occupancy == 0.0) | (square_occupancy == 1.0))

    # In initial state, ranks 3, 4, 5, 6 (index 2, 3, 4, 5) are empty
    assert np.all(square_occupancy[2:6, :] == 0.0)


def test_side_to_move_polarity_invariant():
    encoder = StandardTensorEncoder()
    state_w = GameState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    state_b = GameState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")

    tensor_w = encoder.encode(state_w)
    tensor_b = encoder.encode(state_b)

    assert np.all(tensor_w[16] == 1.0)
    assert np.all(tensor_b[16] == -1.0)


def test_castling_rights_invariants():
    encoder = StandardTensorEncoder()
    # All rights active
    state_all = GameState.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    t_all = encoder.encode(state_all)
    assert np.all(t_all[12] == 1.0)
    assert np.all(t_all[13] == 1.0)
    assert np.all(t_all[14] == 1.0)
    assert np.all(t_all[15] == 1.0)

    # Only White Kingside and Black Queenside
    state_subset = GameState.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
    t_subset = encoder.encode(state_subset)
    assert np.all(t_subset[12] == 1.0)
    assert np.all(t_subset[13] == 0.0)
    assert np.all(t_subset[14] == 0.0)
    assert np.all(t_subset[15] == 1.0)


def test_en_passant_spatial_plane_invariant():
    encoder = StandardTensorEncoder()
    # En passant target on e6 (rank 5, file 4)
    state_ep = GameState.from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
    tensor = encoder.encode(state_ep)

    # Plane 17 must have exactly one 1.0 bit at (rank=5, file=4)
    assert tensor[17].sum() == 1.0
    ep_file, ep_rank = square_to_coords(algebraic_to_square("e6"))
    assert tensor[17, ep_rank, ep_file] == 1.0

    # When no EP target exists, Plane 17 must be entirely 0.0
    state_no_ep = GameState.initial()
    tensor_no_ep = encoder.encode(state_no_ep)
    assert tensor_no_ep[17].sum() == 0.0


def test_promotion_tensor_invariant():
    encoder = StandardTensorEncoder()
    # White pawn promotes on e8 to Knight
    state = GameState.from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
    n_promo = [m for m in MoveGenerator.generate_legal_moves(state) if m.to_uci() == "e7e8n"][0]
    next_state = MoveGenerator.apply_move(state, n_promo)

    tensor = encoder.encode(next_state)
    e8_file, e8_rank = square_to_coords(algebraic_to_square("e8"))
    
    # White Pawns plane (0) must now have 0 pawns
    assert tensor[0].sum() == 0.0
    # White Knights plane (1) must have 1 knight at e8
    assert tensor[1, e8_rank, e8_file] == 1.0
