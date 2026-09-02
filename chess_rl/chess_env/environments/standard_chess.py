"""
Standard 8x8 Chess Environment implementation for Reinforcement Learning.
Connects GameState, MoveGenerator, ActionEncoder, and TensorEncoder to BaseChessEnvironment.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from chess_rl.chess_env.base import BaseChessEnvironment, Player, StepResult
from chess_rl.chess_env.board.types import Color
from chess_rl.chess_env.board.move import Move
from chess_rl.chess_env.board.action_space import ActionEncoder, ACTION_SPACE_SIZE
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator
from chess_rl.chess_env.representation.tensor_encoder import StandardTensorEncoder
from chess_rl.config.schema import EnvironmentConfig
from chess_rl.chess_env.environments.registry import EnvironmentRegistry


class StandardChessEnvironment(BaseChessEnvironment):
    """
    Standard FIDE 8x8 Chess Environment adhering to Gym-style RL interfaces.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self._config = config or EnvironmentConfig()
        self._encoder = StandardTensorEncoder(board_shape=(8, 8))
        self._action_space_size = ACTION_SPACE_SIZE
        self.state = GameState.initial()

    @property
    def name(self) -> str:
        return "standard_8x8_chess"

    @property
    def board_shape(self) -> Tuple[int, int]:
        return (8, 8)

    @property
    def num_channels(self) -> int:
        return self._encoder.num_channels

    @property
    def action_space_size(self) -> int:
        return self._action_space_size

    @property
    def current_player(self) -> Player:
        return Player.WHITE if self.state.side_to_move == Color.WHITE else Player.BLACK

    def reset(self, seed: Optional[int] = None) -> StepResult:
        """Reset environment to initial starting position."""
        self.state = GameState.initial()
        legal_moves = self.legal_actions()
        return StepResult(
            observation=self.get_observation_tensor(),
            reward=0.0,
            is_terminal=False,
            current_player=self.current_player,
            legal_actions=legal_moves,
            info={"fen": self.state.to_fen()},
        )

    def step(self, action: Move) -> StepResult:
        """
        Execute a legal Move in the environment.
        """
        legal_moves = self.legal_actions()
        if action not in legal_moves:
            raise ValueError(f"Illegal move attempted: {action.to_uci()} in state: {self.state.to_fen()}")

        self.state = MoveGenerator.apply_move(self.state, action)
        terminal = self.is_terminal()
        reward = self.get_reward(self.current_player) if terminal else 0.0

        return StepResult(
            observation=self.get_observation_tensor(),
            reward=reward,
            is_terminal=terminal,
            current_player=self.current_player,
            legal_actions=self.legal_actions(),
            info={
                "fen": self.state.to_fen(),
                "is_check": MoveGenerator.is_in_check(self.state.side_to_move, self.state),
                "is_checkmate": MoveGenerator.is_checkmate(self.state),
                "is_stalemate": MoveGenerator.is_stalemate(self.state),
                "is_draw_fifty": MoveGenerator.is_fifty_move_draw(self.state),
                "is_draw_repetition": MoveGenerator.is_threefold_repetition(self.state),
            },
        )

    def legal_actions(self) -> List[Move]:
        """Return all legal moves from the current position."""
        return MoveGenerator.generate_legal_moves(self.state)

    def get_legal_action_mask(self) -> np.ndarray:
        """Return boolean 1D mask of legal actions."""
        return ActionEncoder.create_legal_mask(self.legal_actions(), self._action_space_size)

    def get_observation_tensor(self) -> np.ndarray:
        """Return (19, 8, 8) observation tensor."""
        return self._encoder.encode(self.state)

    def is_terminal(self) -> bool:
        """Check if game has reached a terminal state."""
        return MoveGenerator.is_game_over(self.state)

    def get_reward(self, player: Player) -> float:
        """
        Return terminal game outcome from perspective of specified player.
        +1.0: Win
        -1.0: Loss
         0.0: Draw
        """
        target_color = Color.WHITE if player == Player.WHITE else Color.BLACK

        if MoveGenerator.is_checkmate(self.state):
            # Side to move was mated (lost)
            loser = self.state.side_to_move
            return 1.0 if target_color != loser else -1.0

        # Draw by stalemate, 50-move, repetition, or insufficient material
        return 0.0

    def copy(self) -> "StandardChessEnvironment":
        """Return deep copy of environment instance."""
        new_env = StandardChessEnvironment(self._config)
        new_env.state = self.state.copy()
        return new_env

    def fen(self) -> str:
        """Return current FEN string."""
        return self.state.to_fen()

    def set_fen(self, fen_str: str) -> None:
        """Load state directly from FEN string."""
        self.state = GameState.from_fen(fen_str)

    def render(self, mode: str = "text") -> Optional[str]:
        """Render board representation."""
        rendered = self.state.render_ascii()
        if mode == "print":
            print(rendered)
            return None
        return rendered


# Auto-register in EnvironmentRegistry
EnvironmentRegistry.register("full_chess_8x8", lambda cfg: StandardChessEnvironment(cfg))
EnvironmentRegistry.register("standard_8x8", lambda cfg: StandardChessEnvironment(cfg))
