"""
Chess environment and rules engine package for chess_rl.
"""

from chess_rl.chess_env.base import (
    BaseChessEnvironment,
    Player,
    Move,
    StepResult,
)
from chess_rl.chess_env.board.types import (
    Color,
    PieceType,
    Piece,
    CastlingRights,
    square_to_coords,
    coords_to_square,
    square_to_algebraic,
    algebraic_to_square,
)
from chess_rl.chess_env.board.action_space import (
    ActionEncoder,
    ACTION_SPACE_SIZE,
)
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator
from chess_rl.chess_env.representation.tensor_encoder import (
    BaseTensorEncoder,
    StandardTensorEncoder,
)
from chess_rl.chess_env.environments.registry import EnvironmentRegistry
from chess_rl.chess_env.environments.standard_chess import StandardChessEnvironment

__all__ = [
    "BaseChessEnvironment",
    "Player",
    "Move",
    "StepResult",
    "Color",
    "PieceType",
    "Piece",
    "CastlingRights",
    "square_to_coords",
    "coords_to_square",
    "square_to_algebraic",
    "algebraic_to_square",
    "ActionEncoder",
    "ACTION_SPACE_SIZE",
    "GameState",
    "MoveGenerator",
    "BaseTensorEncoder",
    "StandardTensorEncoder",
    "EnvironmentRegistry",
    "StandardChessEnvironment",
]
