"""
Board state tensor encoding for neural network input.

Feature Planes Specification (Shape: 19 x 8 x 8):
- Plane 0:  White Pawns
- Plane 1:  White Knights
- Plane 2:  White Bishops
- Plane 3:  White Rooks
- Plane 4:  White Queens
- Plane 5:  White King
- Plane 6:  Black Pawns
- Plane 7:  Black Knights
- Plane 8:  Black Bishops
- Plane 9:  Black Rooks
- Plane 10: Black Queens
- Plane 11: Black King
- Plane 12: White Kingside Castling Right (1.0 if available, 0.0 otherwise)
- Plane 13: White Queenside Castling Right (1.0 if available, 0.0 otherwise)
- Plane 14: Black Kingside Castling Right (1.0 if available, 0.0 otherwise)
- Plane 15: Black Queenside Castling Right (1.0 if available, 0.0 otherwise)
- Plane 16: Active Player Turn (+1.0 for White, -1.0 for Black)
- Plane 17: En Passant Target Square (1.0 at target (rank, file), 0.0 otherwise)
- Plane 18: Normalized Halfmove Clock (halfmove_clock / 100.0)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np

from chess_rl.chess_env.board.types import Color, PieceType, square_to_coords
from chess_rl.chess_env.rules.game_state import GameState


class BaseTensorEncoder(ABC):
    """Abstract Base Class for converting board states to neural network tensor inputs."""

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Total feature planes encoded."""
        pass

    @property
    @abstractmethod
    def board_shape(self) -> Tuple[int, int]:
        """Expected spatial dimensions (H, W)."""
        pass

    @abstractmethod
    def encode(self, state: GameState) -> np.ndarray:
        """
        Encode GameState into float32 array of shape (C, H, W).
        """
        pass


class StandardTensorEncoder(BaseTensorEncoder):
    """
    Standard 19-plane tensor encoder for 8x8 chess positions.
    """

    def __init__(self, board_shape: Tuple[int, int] = (8, 8)):
        self._board_shape = board_shape
        self._num_channels = 19

        # Piece to channel index mapping
        self._piece_channel_map = {
            (Color.WHITE, PieceType.PAWN): 0,
            (Color.WHITE, PieceType.KNIGHT): 1,
            (Color.WHITE, PieceType.BISHOP): 2,
            (Color.WHITE, PieceType.ROOK): 3,
            (Color.WHITE, PieceType.QUEEN): 4,
            (Color.WHITE, PieceType.KING): 5,
            (Color.BLACK, PieceType.PAWN): 6,
            (Color.BLACK, PieceType.KNIGHT): 7,
            (Color.BLACK, PieceType.BISHOP): 8,
            (Color.BLACK, PieceType.ROOK): 9,
            (Color.BLACK, PieceType.QUEEN): 10,
            (Color.BLACK, PieceType.KING): 11,
        }

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def board_shape(self) -> Tuple[int, int]:
        return self._board_shape

    def encode(self, state: Any) -> np.ndarray:
        """
        Encode a GameState instance into an exact (19, 8, 8) float32 tensor.
        """
        tensor = np.zeros(
            (self._num_channels, self._board_shape[0], self._board_shape[1]),
            dtype=np.float32,
        )

        if not isinstance(state, GameState):
            return tensor

        # 1. Populate Piece Placement Planes (0..11)
        for sq in range(64):
            piece = state.board[sq]
            if piece is not None:
                ch = self._piece_channel_map.get((piece.color, piece.piece_type))
                if ch is not None:
                    file, rank = square_to_coords(sq)
                    tensor[ch, rank, file] = 1.0

        # 2. Populate Castling Rights (12..15)
        if state.castling_rights.white_kingside:
            tensor[12, :, :] = 1.0
        if state.castling_rights.white_queenside:
            tensor[13, :, :] = 1.0
        if state.castling_rights.black_kingside:
            tensor[14, :, :] = 1.0
        if state.castling_rights.black_queenside:
            tensor[15, :, :] = 1.0

        # 3. Active Turn Plane (16)
        tensor[16, :, :] = 1.0 if state.side_to_move == Color.WHITE else -1.0

        # 4. En Passant Target Square Plane (17)
        if state.en_passant_target is not None:
            ep_file, ep_rank = square_to_coords(state.en_passant_target)
            tensor[17, ep_rank, ep_file] = 1.0

        # 5. Normalized Halfmove Clock (18)
        tensor[18, :, :] = min(state.halfmove_clock / 100.0, 1.0)

        return tensor
