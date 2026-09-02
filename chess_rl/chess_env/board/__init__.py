"""
Board representation, types, moves, and action space encoding.
"""

from chess_rl.chess_env.board.types import (
    Color,
    PieceType,
    Piece,
    CastlingRights,
    SQUARES,
    square_to_coords,
    coords_to_square,
    square_to_algebraic,
    algebraic_to_square,
)
from chess_rl.chess_env.board.move import Move
from chess_rl.chess_env.board.action_space import (
    ActionEncoder,
    ACTION_SPACE_SIZE,
)

__all__ = [
    "Color",
    "PieceType",
    "Piece",
    "CastlingRights",
    "SQUARES",
    "square_to_coords",
    "coords_to_square",
    "square_to_algebraic",
    "algebraic_to_square",
    "Move",
    "ActionEncoder",
    "ACTION_SPACE_SIZE",
]
