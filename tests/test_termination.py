"""
Unit tests for game termination conditions: Checkmate, Stalemate, 50-move rule, Repetition, and Insufficient Material.
"""

import pytest
from chess_rl.chess_env.board.types import Color
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator


def test_scholars_mate_and_fools_mate():
    # Fool's mate terminal position: White King on e1 in check by Black Queen on h4 with 0 legal moves
    fools_mate = GameState.from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert MoveGenerator.is_in_check(Color.WHITE, fools_mate)
    assert MoveGenerator.is_checkmate(fools_mate)
    assert not MoveGenerator.is_stalemate(fools_mate)
    assert MoveGenerator.is_game_over(fools_mate)


def test_stalemate():
    # Black King on a8, White Queen on b6, White King on h1. Black is NOT in check, has 0 moves.
    stalemate = GameState.from_fen("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    assert not MoveGenerator.is_in_check(Color.BLACK, stalemate)
    assert MoveGenerator.is_stalemate(stalemate)
    assert not MoveGenerator.is_checkmate(stalemate)
    assert MoveGenerator.is_game_over(stalemate)


def test_fifty_move_rule():
    state = GameState.from_fen("8/8/8/8/8/8/4k3/4K3 w - - 100 50")
    assert MoveGenerator.is_fifty_move_draw(state)
    assert MoveGenerator.is_game_over(state)


def test_threefold_repetition():
    state = GameState.initial()
    # Move knight back and forth: Nf3, Nf6, Ng1, Ng8, Nf3, Nf6, Ng1, Ng8
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
    for m_uci in moves:
        m = [m for m in MoveGenerator.generate_legal_moves(state) if m.to_uci() == m_uci][0]
        state = MoveGenerator.apply_move(state, m)

    assert MoveGenerator.is_threefold_repetition(state)
    assert MoveGenerator.is_game_over(state)


def test_insufficient_material():
    # King vs King
    kvk = GameState.from_fen("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
    assert MoveGenerator.is_insufficient_material(kvk)
    assert MoveGenerator.is_game_over(kvk)

    # King + Bishop vs King
    kbvk = GameState.from_fen("8/8/8/4k3/8/5B2/8/4K3 w - - 0 1")
    assert MoveGenerator.is_insufficient_material(kbvk)

    # King + Knight vs King
    knvk = GameState.from_fen("8/8/8/4k3/8/5N2/8/4K3 w - - 0 1")
    assert MoveGenerator.is_insufficient_material(knvk)

    # King + Pawn vs King (Sufficient material!)
    kpvk = GameState.from_fen("8/8/8/4k3/8/5P2/8/4K3 w - - 0 1")
    assert not MoveGenerator.is_insufficient_material(kpvk)
