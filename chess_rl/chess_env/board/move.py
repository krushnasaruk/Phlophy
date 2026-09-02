"""
Typed Move representation with UCI formatting and parsing.
"""

from dataclasses import dataclass
from typing import Optional
from chess_rl.chess_env.board.types import (
    PieceType,
    square_to_algebraic,
    algebraic_to_square,
)


@dataclass(frozen=True)
class Move:
    """
    Immutable representation of a discrete chess move.
    """
    from_square: int
    to_square: int
    promotion_piece: Optional[PieceType] = None
    is_castling: bool = False
    is_en_passant: bool = False

    def to_uci(self) -> str:
        """Convert move to UCI coordinate string (e.g. 'e2e4', 'e7e8q')."""
        from_alg = square_to_algebraic(self.from_square)
        to_alg = square_to_algebraic(self.to_square)
        promo = ""
        if self.promotion_piece is not None:
            promo_chars = {
                PieceType.QUEEN: "q",
                PieceType.ROOK: "r",
                PieceType.BISHOP: "b",
                PieceType.KNIGHT: "n",
            }
            promo = promo_chars.get(self.promotion_piece, "q")
        return f"{from_alg}{to_alg}{promo}"

    @classmethod
    def from_uci(cls, uci_str: str, is_castling: bool = False, is_en_passant: bool = False) -> "Move":
        """Parse UCI coordinate string into Move instance."""
        if len(uci_str) < 4 or len(uci_str) > 5:
            raise ValueError(f"Invalid UCI move string: '{uci_str}'")
        from_sq = algebraic_to_square(uci_str[:2])
        to_sq = algebraic_to_square(uci_str[2:4])
        promo = None
        if len(uci_str) == 5:
            promo_char = uci_str[4].lower()
            promo_map = {
                "q": PieceType.QUEEN,
                "r": PieceType.ROOK,
                "b": PieceType.BISHOP,
                "n": PieceType.KNIGHT,
            }
            if promo_char not in promo_map:
                raise ValueError(f"Invalid promotion character in UCI: '{promo_char}'")
            promo = promo_map[promo_char]

        return cls(
            from_square=from_sq,
            to_square=to_sq,
            promotion_piece=promo,
            is_castling=is_castling,
            is_en_passant=is_en_passant,
        )

    def to_action_index(self, board_size: int = 8) -> int:
        """
        Encode move into discrete action index.
        Normal moves & Queen promo: from_sq * 64 + to_sq in [0, 4095].
        Underpromotions: mapped beyond 4096.
        """
        base_idx = self.from_square * 64 + self.to_square
        if self.promotion_piece is None or self.promotion_piece == PieceType.QUEEN:
            return base_idx
        
        # Underpromotion offset encoding:
        # Knight (+4096), Bishop (+4096 + 64), Rook (+4096 + 128)
        # Specifically: from_file (0..7) -> to_file (delta: -1, 0, +1)
        promo_offset = {
            PieceType.KNIGHT: 4096,
            PieceType.BISHOP: 4096 + 64 * 64,
            PieceType.ROOK: 4096 + 2 * 64 * 64,
        }
        return promo_offset[self.promotion_piece] + base_idx

    def __str__(self) -> str:
        return self.to_uci()

    def __repr__(self) -> str:
        promo_str = f", promo={self.promotion_piece.name}" if self.promotion_piece else ""
        flags = ""
        if self.is_castling:
            flags += ", castling=True"
        if self.is_en_passant:
            flags += ", ep=True"
        return f"Move({square_to_algebraic(self.from_square)}->{square_to_algebraic(self.to_square)}{promo_str}{flags})"
