"""
Chess environment subpackage and registry.
"""

from chess_rl.chess_env.environments.registry import EnvironmentRegistry
from chess_rl.chess_env.environments.standard_chess import StandardChessEnvironment

__all__ = [
    "EnvironmentRegistry",
    "StandardChessEnvironment",
]
