"""
Discrete Action Space encoding, decoding, and legal action masking.

Action Space Specification (Total Size = 4240):
- Normal Moves & Queen Promotions (0 .. 4095): from_square * 64 + to_square
- Underpromotions (4096 .. 4227):
  - Knight underpromotions (4096 .. 4139): 44 slots
  - Bishop underpromotions (4140 .. 4183): 44 slots
  - Rook underpromotions (4184 .. 4227): 44 slots
- Unused padding: 4228 .. 4239 (safe buffer to align to 4240)
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from chess_rl.chess_env.board.types import PieceType, square_to_coords, coords_to_square
from chess_rl.chess_env.board.move import Move

ACTION_SPACE_SIZE = 4240


class ActionEncoder:
    """
    Bijective and collision-free mapping between Move instances and discrete action indices.
    """

    _PROMO_MOVES: List[Tuple[int, int]] = []
    _PROMO_LOOKUP: Dict[Tuple[int, int], int] = {}
    _INITIALIZED: bool = False

    @classmethod
    def _init_promo_tables(cls):
        if cls._INITIALIZED:
            return
        cls._PROMO_MOVES.clear()
        cls._PROMO_LOOKUP.clear()

        # White promotions: rank 7 (squares 48..55) to rank 8 (squares 56..63)
        for file in range(8):
            from_sq = coords_to_square(file, 6)  # Rank 7
            for d_file in [-1, 0, 1]:
                target_file = file + d_file
                if 0 <= target_file < 8:
                    to_sq = coords_to_square(target_file, 7)  # Rank 8
                    cls._PROMO_LOOKUP[(from_sq, to_sq)] = len(cls._PROMO_MOVES)
                    cls._PROMO_MOVES.append((from_sq, to_sq))

        # Black promotions: rank 2 (squares 8..15) to rank 1 (squares 0..7)
        for file in range(8):
            from_sq = coords_to_square(file, 1)  # Rank 2
            for d_file in [-1, 0, 1]:
                target_file = file + d_file
                if 0 <= target_file < 8:
                    to_sq = coords_to_square(target_file, 0)  # Rank 1
                    cls._PROMO_LOOKUP[(from_sq, to_sq)] = len(cls._PROMO_MOVES)
                    cls._PROMO_MOVES.append((from_sq, to_sq))

        cls._INITIALIZED = True

    @classmethod
    def encode(cls, move: Move) -> int:
        """
        Map a Move instance to its unique action index in [0, ACTION_SPACE_SIZE - 1].
        """
        cls._init_promo_tables()
        if not (0 <= move.from_square < 64 and 0 <= move.to_square < 64):
            raise ValueError(f"Move squares out of bounds: from={move.from_square}, to={move.to_square}")

        base_idx = move.from_square * 64 + move.to_square

        # Normal move or Queen promotion
        if move.promotion_piece is None or move.promotion_piece == PieceType.QUEEN:
            return base_idx

        # Underpromotions
        promo_key = (move.from_square, move.to_square)
        if promo_key not in cls._PROMO_LOOKUP:
            raise ValueError(f"Invalid underpromotion coordinates: {move}")

        promo_idx = cls._PROMO_LOOKUP[promo_key]
        num_promos_per_piece = len(cls._PROMO_MOVES)  # 44

        if move.promotion_piece == PieceType.KNIGHT:
            action_id = 4096 + promo_idx
        elif move.promotion_piece == PieceType.BISHOP:
            action_id = 4096 + num_promos_per_piece + promo_idx
        elif move.promotion_piece == PieceType.ROOK:
            action_id = 4096 + 2 * num_promos_per_piece + promo_idx
        else:
            action_id = base_idx

        if not (0 <= action_id < ACTION_SPACE_SIZE):
            raise ValueError(f"Encoded action index {action_id} exceeds bounds [0, {ACTION_SPACE_SIZE - 1}]")

        return action_id

    @classmethod
    def decode(cls, action_idx: int, state: Optional[Any] = None) -> Move:
        """
        Reconstruct a Move instance from its action index.
        If state is provided, correctly assigns PieceType.QUEEN for pawn promotions.
        """
        cls._init_promo_tables()
        if not (0 <= action_idx < ACTION_SPACE_SIZE):
            raise ValueError(f"Action index {action_idx} out of valid bounds [0, {ACTION_SPACE_SIZE - 1}]")

        num_promos_per_piece = len(cls._PROMO_MOVES)  # 44
        max_underpromo_idx = 4096 + 3 * num_promos_per_piece  # 4228

        if action_idx >= max_underpromo_idx:
            raise ValueError(f"Action index {action_idx} is in unused padding range [{max_underpromo_idx}, {ACTION_SPACE_SIZE - 1}]")

        if action_idx < 4096:
            from_sq = action_idx // 64
            to_sq = action_idx % 64
            promo = None

            if state is not None:
                piece = state.board[from_sq]
                if piece is not None and piece.piece_type == PieceType.PAWN:
                    _, to_rank = square_to_coords(to_sq)
                    if to_rank == 0 or to_rank == 7:
                        promo = PieceType.QUEEN
            else:
                # Default heuristic: if from_rank is 7th/2nd and to_rank is 8th/1st
                from_file, from_rank = square_to_coords(from_sq)
                to_file, to_rank = square_to_coords(to_sq)
                if (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0):
                    promo = PieceType.QUEEN

            return Move(from_square=from_sq, to_square=to_sq, promotion_piece=promo)

        underpromo_idx = action_idx - 4096
        piece_type_idx = underpromo_idx // num_promos_per_piece
        slot_idx = underpromo_idx % num_promos_per_piece

        from_sq, to_sq = cls._PROMO_MOVES[slot_idx]
        promo_types = [PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK]
        promo = promo_types[min(piece_type_idx, 2)]

        return Move(from_square=from_sq, to_square=to_sq, promotion_piece=promo)

    @classmethod
    def create_legal_mask(
        cls, legal_moves: List[Move], action_space_size: int = ACTION_SPACE_SIZE
    ) -> np.ndarray:
        """
        Generate a boolean 1D numpy array of shape (action_space_size,) where True=legal.
        """
        mask = np.zeros(action_space_size, dtype=bool)
        for move in legal_moves:
            idx = cls.encode(move)
            if idx < action_space_size:
                mask[idx] = True
        return mask
