"""
Environment registry and factory functions.
Allows dynamic instantiation of chess environments from configuration schemas.
"""

from typing import Dict, Type, Callable, Any
from chess_rl.chess_env.base import BaseChessEnvironment
from chess_rl.config.schema import EnvironmentConfig


class EnvironmentRegistry:
    """Registry for chess environment variations."""
    _registry: Dict[str, Callable[[EnvironmentConfig], BaseChessEnvironment]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[EnvironmentConfig], BaseChessEnvironment]) -> None:
        """Register an environment factory."""
        cls._registry[name] = factory

    @classmethod
    def create(cls, config: EnvironmentConfig) -> BaseChessEnvironment:
        """Instantiate an environment based on EnvironmentConfig."""
        if config.name not in cls._registry and config.board_type not in cls._registry:
            # Fallback or raise
            registered_keys = list(cls._registry.keys())
            raise KeyError(
                f"Environment '{config.name}' (type: '{config.board_type}') not found in registry. "
                f"Available: {registered_keys}"
            )
        factory = cls._registry.get(config.name) or cls._registry.get(config.board_type)
        return factory(config)

    @classmethod
    def list_available(cls) -> list:
        return list(cls._registry.keys())
