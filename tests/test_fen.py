"""
Unit tests for Forsyth-Edwards Notation (FEN) parsing and serialization.
"""

import pytest
from chess_rl.chess_env.rules.game_state import GameState


FEN_TEST_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "k7/8/1Q6/8/8/8/8/7K b - - 0 1",
]


def test_fen_roundtrips():
    for fen in FEN_TEST_POSITIONS:
        state = GameState.from_fen(fen)
        generated_fen = state.to_fen()
        assert generated_fen == fen

        # State equality across roundtrip
        state2 = GameState.from_fen(generated_fen)
        assert state == state2


def test_fen_invalid_strings():
    with pytest.raises(ValueError):
        GameState.from_fen("invalid fen string")

    # Invalid rank count
    with pytest.raises(ValueError):
        GameState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1")
