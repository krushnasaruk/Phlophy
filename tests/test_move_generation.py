"""
Unit tests for move generation, attack detection, check detection, and pin resolution.
"""

import pytest
from chess_rl.chess_env.board.types import (
    Color,
    PieceType,
    Piece,
    algebraic_to_square,
)
from chess_rl.chess_env.board.move import Move
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator


def test_initial_position_legal_moves():
    state = GameState.initial()
    moves = MoveGenerator.generate_legal_moves(state)
    assert len(moves) == 20

    # 16 pawn moves (8 single pushes, 8 double pushes) + 4 knight hops
    pawn_moves = [m for m in moves if state.board[m.from_square].piece_type == PieceType.PAWN]
    knight_moves = [m for m in moves if state.board[m.from_square].piece_type == PieceType.KNIGHT]

    assert len(pawn_moves) == 16
    assert len(knight_moves) == 4


def test_attack_detection_all_pieces():
    # Setup custom position: White pieces attacking e4 (sq 28)
    state = GameState()
    e4 = algebraic_to_square("e4")

    # 1. Pawn attack from d3 (sq 19)
    d3 = algebraic_to_square("d3")
    state.set_piece(d3, Piece(PieceType.PAWN, Color.WHITE))
    assert MoveGenerator.is_square_attacked(e4, Color.WHITE, state)
    assert not MoveGenerator.is_square_attacked(e4, Color.BLACK, state)
    state.set_piece(d3, None)

    # 2. Knight attack from c3 (sq 18)
    c3 = algebraic_to_square("c3")
    state.set_piece(c3, Piece(PieceType.KNIGHT, Color.WHITE))
    assert MoveGenerator.is_square_attacked(e4, Color.WHITE, state)
    state.set_piece(c3, None)

    # 3. Bishop attack from b1 (sq 1)
    b1 = algebraic_to_square("b1")
    state.set_piece(b1, Piece(PieceType.BISHOP, Color.WHITE))
    assert MoveGenerator.is_square_attacked(e4, Color.WHITE, state)
    # Block diagonal with pawn on c2 (sq 10)
    c2 = algebraic_to_square("c2")
    state.set_piece(c2, Piece(PieceType.PAWN, Color.BLACK))
    assert not MoveGenerator.is_square_attacked(e4, Color.WHITE, state)
    state.set_piece(b1, None)
    state.set_piece(c2, None)

    # 4. Rook attack from e1 (sq 4)
    e1 = algebraic_to_square("e1")
    state.set_piece(e1, Piece(PieceType.ROOK, Color.WHITE))
    assert MoveGenerator.is_square_attacked(e4, Color.WHITE, state)
    # Block with piece on e2 (sq 12)
    e2 = algebraic_to_square("e2")
    state.set_piece(e2, Piece(PieceType.PAWN, Color.WHITE))
    assert not MoveGenerator.is_square_attacked(e4, Color.WHITE, state)
    state.set_piece(e1, None)
    state.set_piece(e2, None)

    # 5. King attack from e3 (sq 20)
    e3 = algebraic_to_square("e3")
    state.set_piece(e3, Piece(PieceType.KING, Color.WHITE))
    assert MoveGenerator.is_square_attacked(e4, Color.WHITE, state)


def test_check_detection_and_resolution():
    # Position: White King on e1, Black Rook on e8 -> White is in check
    state = GameState.from_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert MoveGenerator.is_in_check(Color.WHITE, state)
    assert not MoveGenerator.is_in_check(Color.BLACK, state)

    legal_moves = MoveGenerator.generate_legal_moves(state)
    # King must move out of check (d1, f1, d2, f2 are safe; e2 is on the e-file ray so still in check!)
    legal_uci = [m.to_uci() for m in legal_moves]
    assert "e1d1" in legal_uci
    assert "e1f1" in legal_uci
    assert "e1d2" in legal_uci
    assert "e1f2" in legal_uci
    assert "e1e2" not in legal_uci  # Illegal: stays on ray


def test_absolute_pin():
    # Position: White King on e1, White Knight on e2, Black Rook on e8
    # Knight on e2 is absolutely pinned to the King on e1
    state = GameState.from_fen("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1")
    assert not MoveGenerator.is_in_check(Color.WHITE, state)

    legal_moves = MoveGenerator.generate_legal_moves(state)
    # Pinned Knight cannot move at all
    knight_moves = [m for m in legal_moves if m.from_square == algebraic_to_square("e2")]
    assert len(knight_moves) == 0

    # King can step off the e-file
    assert len(legal_moves) > 0
