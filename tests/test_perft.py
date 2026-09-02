"""
Unit tests for PERFT (Performance / Move Path Enumeration) validation.
Validates legal move generation correctness against canonical theoretical node counts.
"""

import pytest
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.perft import perft


def test_perft_initial_position_depth_1_to_4():
    """
    Standard Starting Position PERFT Validation:
    Depth 1 -> 20
    Depth 2 -> 400
    Depth 3 -> 8,902
    Depth 4 -> 197,281
    """
    state = GameState.initial()

    assert perft(state, 1) == 20
    assert perft(state, 2) == 400
    assert perft(state, 3) == 8902
    assert perft(state, 4) == 197281


def test_perft_kiwipete_position():
    """
    Position 2 (Kiwipete): Stresses complex castling, en passant, pins, and checks.
    FEN: r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
    """
    state = GameState.from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")

    assert perft(state, 1) == 48
    assert perft(state, 2) == 2039


def test_perft_endgame_pins():
    """
    Position 3: Stresses endgame rook and pawn pins.
    FEN: 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1
    """
    state = GameState.from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")

    assert perft(state, 1) == 14
    assert perft(state, 2) == 191
    assert perft(state, 3) == 2812


def test_perft_promotions_and_checks():
    """
    Position 4: Stresses simultaneous promotions and check evasions.
    FEN: r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
    """
    state = GameState.from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")

    assert perft(state, 1) == 6
    assert perft(state, 2) == 264


def test_perft_promotions_and_discovered_checks():
    """
    Position 5: Stresses double promotions and discovered checks.
    FEN: n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1
    """
    state = GameState.from_fen("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1")

    assert perft(state, 1) == 24
    assert perft(state, 2) == 496
