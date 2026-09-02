"""
Unit tests for coordinate systems, piece representations, Move objects, and action encodings.
"""

import pytest
import numpy as np

from chess_rl.chess_env.board.types import (
    Color,
    PieceType,
    Piece,
    CastlingRights,
    square_to_algebraic,
    algebraic_to_square,
    square_to_coords,
    coords_to_square,
    E1, E4, E8, A1, H8,
)
from chess_rl.chess_env.board.move import Move
from chess_rl.chess_env.board.action_space import ActionEncoder, ACTION_SPACE_SIZE


def test_coordinate_conversions():
    assert square_to_algebraic(0) == "a1"
    assert square_to_algebraic(7) == "h1"
    assert square_to_algebraic(8) == "a2"
    assert square_to_algebraic(28) == "e4"
    assert square_to_algebraic(63) == "h8"

    assert algebraic_to_square("a1") == 0
    assert algebraic_to_square("h1") == 7
    assert algebraic_to_square("e4") == 28
    assert algebraic_to_square("h8") == 63

    for sq in range(64):
        file, rank = square_to_coords(sq)
        assert 0 <= file < 8
        assert 0 <= rank < 8
        assert coords_to_square(file, rank) == sq
        alg = square_to_algebraic(sq)
        assert algebraic_to_square(alg) == sq


def test_piece_symbols():
    wp = Piece(PieceType.PAWN, Color.WHITE)
    bp = Piece(PieceType.PAWN, Color.BLACK)
    wq = Piece(PieceType.QUEEN, Color.WHITE)
    bk = Piece(PieceType.KING, Color.BLACK)

    assert wp.symbol == "P"
    assert bp.symbol == "p"
    assert wq.symbol == "Q"
    assert bk.symbol == "k"

    pt, col = PieceType.from_symbol("R")
    assert pt == PieceType.ROOK and col == Color.WHITE
    pt, col = PieceType.from_symbol("n")
    assert pt == PieceType.KNIGHT and col == Color.BLACK


def test_castling_rights():
    cr = CastlingRights(True, True, True, True)
    assert cr.to_fen() == "KQkq"

    cr_none = CastlingRights(False, False, False, False)
    assert cr_none.to_fen() == "-"

    cr_parsed = CastlingRights.from_fen("Kq")
    assert cr_parsed.white_kingside
    assert not cr_parsed.white_queenside
    assert not cr_parsed.black_kingside
    assert cr_parsed.black_queenside


def test_move_uci_roundtrip():
    m1 = Move(from_square=12, to_square=28)  # e2 -> e4
    assert m1.to_uci() == "e2e4"

    m_parsed = Move.from_uci("e2e4")
    assert m_parsed.from_square == 12
    assert m_parsed.to_square == 28
    assert m_parsed.promotion_piece is None

    m_promo = Move(from_square=52, to_square=60, promotion_piece=PieceType.QUEEN)
    assert m_promo.to_uci() == "e7e8q"
    m_promo_parsed = Move.from_uci("e7e8q")
    assert m_promo_parsed.promotion_piece == PieceType.QUEEN


def test_action_space_encoder_decoder():
    m = Move(from_square=12, to_square=28)  # e2e4
    idx = ActionEncoder.encode(m)
    assert 0 <= idx < ACTION_SPACE_SIZE
    decoded = ActionEncoder.decode(idx)
    assert decoded.from_square == m.from_square
    assert decoded.to_square == m.to_square

    # Underpromotion
    m_underpromo = Move(from_square=52, to_square=60, promotion_piece=PieceType.KNIGHT)
    idx_up = ActionEncoder.encode(m_underpromo)
    assert idx_up >= 4096
    decoded_up = ActionEncoder.decode(idx_up)
    assert decoded_up.from_square == 52
    assert decoded_up.to_square == 60
    assert decoded_up.promotion_piece == PieceType.KNIGHT

    mask = ActionEncoder.create_legal_mask([m, m_underpromo], ACTION_SPACE_SIZE)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask[idx]
    assert mask[idx_up]
    assert not mask[0]  # a1a1 is illegal
