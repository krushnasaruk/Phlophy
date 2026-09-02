"""
Unit tests for special chess rules: Castling, En Passant, and Pawn Promotions.
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


def test_legal_castling():
    # White and Black have clear paths and all rights
    state = GameState.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves = [m.to_uci() for m in MoveGenerator.generate_legal_moves(state)]

    assert "e1g1" in moves  # White kingside
    assert "e1c1" in moves  # White queenside

    # Execute White Kingside castling
    castling_move = [m for m in MoveGenerator.generate_legal_moves(state) if m.to_uci() == "e1g1"][0]
    next_state = MoveGenerator.apply_move(state, castling_move)

    # King on g1, Rook on f1, e1 and h1 empty
    assert next_state.board[algebraic_to_square("g1")].piece_type == PieceType.KING
    assert next_state.board[algebraic_to_square("f1")].piece_type == PieceType.ROOK
    assert next_state.board[algebraic_to_square("e1")] is None
    assert next_state.board[algebraic_to_square("h1")] is None
    assert not next_state.castling_rights.white_kingside
    assert not next_state.castling_rights.white_queenside


def test_illegal_castling_through_check_or_in_check():
    # 1. King in check: Black rook on e8 attacking e1
    state_in_check = GameState.from_fen("4r3/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves = [m.to_uci() for m in MoveGenerator.generate_legal_moves(state_in_check)]
    assert "e1g1" not in moves
    assert "e1c1" not in moves

    # 2. Crossing through check: Black rook on f8 attacking f1
    state_through_check = GameState.from_fen("5r2/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves2 = [m.to_uci() for m in MoveGenerator.generate_legal_moves(state_through_check)]
    assert "e1g1" not in moves2  # f1 is attacked!
    assert "e1c1" in moves2      # queenside is clear and safe


def test_en_passant_execution_and_expiration():
    # Position: White pawn on e5, Black pawn on d7
    state = GameState.from_fen("rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    
    # Black plays d7d5 (double push)
    black_d5 = [m for m in MoveGenerator.generate_legal_moves(state) if m.to_uci() == "d7d5"][0]
    state_after_d5 = MoveGenerator.apply_move(state, black_d5)

    assert state_after_d5.en_passant_target == algebraic_to_square("d6")

    # White can capture en passant e5d6
    legal_white = [m.to_uci() for m in MoveGenerator.generate_legal_moves(state_after_d5)]
    assert "e5d6" in legal_white

    # Execute en passant capture
    ep_move = [m for m in MoveGenerator.generate_legal_moves(state_after_d5) if m.to_uci() == "e5d6"][0]
    state_after_ep = MoveGenerator.apply_move(state_after_d5, ep_move)

    # White pawn is on d6, captured Black pawn on d5 is removed
    assert state_after_ep.board[algebraic_to_square("d6")].piece_type == PieceType.PAWN
    assert state_after_ep.board[algebraic_to_square("d6")].color == Color.WHITE
    assert state_after_ep.board[algebraic_to_square("d5")] is None
    assert state_after_ep.en_passant_target is None


def test_en_passant_pinned_horizontally():
    # Rare chess edge case: White King on e5, White Pawn on f5, Black Pawn on g5, Black Rook on a5
    # Black just played g7g5. If White plays f5xg6 e.p., both pawns vacate the 5th rank,
    # leaving White King exposed to the Black Rook on a5! Thus f5g6 is illegal.
    state = GameState.from_fen("8/8/8/r3KPP1/8/8/8/8 w - g6 0 1")
    legal_moves = [m.to_uci() for m in MoveGenerator.generate_legal_moves(state)]
    assert "f5g6" not in legal_moves


def test_pawn_promotions():
    # White pawn on e7 about to promote to e8
    state = GameState.from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
    legal_moves = [m.to_uci() for m in MoveGenerator.generate_legal_moves(state)]

    assert "e7e8q" in legal_moves
    assert "e7e8r" in legal_moves
    assert "e7e8b" in legal_moves
    assert "e7e8n" in legal_moves

    # Promote to Knight
    n_promo = [m for m in MoveGenerator.generate_legal_moves(state) if m.to_uci() == "e7e8n"][0]
    next_state = MoveGenerator.apply_move(state, n_promo)

    promoted_piece = next_state.board[algebraic_to_square("e8")]
    assert promoted_piece.piece_type == PieceType.KNIGHT
    assert promoted_piece.color == Color.WHITE
