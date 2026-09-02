"""
Canonical coordinate systems, piece types, and core type definitions for chess_rl.

Coordinate System: Little-Endian Rank-File (0 to 63)
  a1 = 0,  b1 = 1,  ..., h1 = 7   (Rank 1)
  a2 = 8,  b2 = 9,  ..., h2 = 15  (Rank 2)
  ...
  a8 = 56, b8 = 57, ..., h8 = 63  (Rank 8)

File: square % 8   (0='a' .. 7='h')
Rank: square // 8  (0='1' .. 7='8')
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


class Color(IntEnum):
    WHITE = 1
    BLACK = -1

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self == Color.WHITE else Color.WHITE

    def __str__(self) -> str:
        return "White" if self == Color.WHITE else "Black"


class PieceType(IntEnum):
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

    def symbol(self, color: Color = Color.WHITE) -> str:
        symbols = {
            PieceType.PAWN: "P",
            PieceType.KNIGHT: "N",
            PieceType.BISHOP: "B",
            PieceType.ROOK: "R",
            PieceType.QUEEN: "Q",
            PieceType.KING: "K",
        }
        sym = symbols[self]
        return sym.upper() if color == Color.WHITE else sym.lower()

    @classmethod
    def from_symbol(cls, char: str) -> Tuple["PieceType", Color]:
        mapping = {
            "p": (cls.PAWN, Color.BLACK),
            "n": (cls.KNIGHT, Color.BLACK),
            "b": (cls.BISHOP, Color.BLACK),
            "r": (cls.ROOK, Color.BLACK),
            "q": (cls.QUEEN, Color.BLACK),
            "k": (cls.KING, Color.BLACK),
            "P": (cls.PAWN, Color.WHITE),
            "N": (cls.KNIGHT, Color.WHITE),
            "B": (cls.BISHOP, Color.WHITE),
            "R": (cls.ROOK, Color.WHITE),
            "Q": (cls.QUEEN, Color.WHITE),
            "K": (cls.KING, Color.WHITE),
        }
        if char not in mapping:
            raise ValueError(f"Invalid piece symbol: '{char}'")
        return mapping[char]


@dataclass(frozen=True)
class Piece:
    piece_type: PieceType
    color: Color

    @property
    def symbol(self) -> str:
        return self.piece_type.symbol(self.color)

    def __str__(self) -> str:
        return self.symbol


@dataclass
class CastlingRights:
    white_kingside: bool = True
    white_queenside: bool = True
    black_kingside: bool = True
    black_queenside: bool = True

    def copy(self) -> "CastlingRights":
        return CastlingRights(
            white_kingside=self.white_kingside,
            white_queenside=self.white_queenside,
            black_kingside=self.black_kingside,
            black_queenside=self.black_queenside,
        )

    def to_fen(self) -> str:
        res = ""
        if self.white_kingside:
            res += "K"
        if self.white_queenside:
            res += "Q"
        if self.black_kingside:
            res += "k"
        if self.black_queenside:
            res += "q"
        return res if res else "-"

    @classmethod
    def from_fen(cls, fen_str: str) -> "CastlingRights":
        if fen_str == "-":
            return cls(False, False, False, False)
        return cls(
            white_kingside="K" in fen_str,
            white_queenside="Q" in fen_str,
            black_kingside="k" in fen_str,
            black_queenside="q" in fen_str,
        )


# Canonical Square Constants
SQUARES = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
] = list(range(64))


def square_to_coords(square: int) -> Tuple[int, int]:
    """Return (file, rank) with 0-indexing (0..7, 0..7)."""
    if not 0 <= square < 64:
        raise ValueError(f"Square {square} out of valid bounds [0, 63]")
    return square % 8, square // 8


def coords_to_square(file: int, rank: int) -> int:
    """Convert 0-indexed file and rank to square index [0..63]."""
    if not (0 <= file < 8 and 0 <= rank < 8):
        raise ValueError(f"Coords ({file}, {rank}) out of valid bounds [0..7, 0..7]")
    return rank * 8 + file


def square_to_algebraic(square: int) -> str:
    """Convert square index to algebraic notation (e.g. 0 -> 'a1', 28 -> 'e4')."""
    file, rank = square_to_coords(square)
    return f"{chr(ord('a') + file)}{rank + 1}"


def algebraic_to_square(alg: str) -> int:
    """Convert algebraic notation to square index (e.g. 'a1' -> 0, 'e4' -> 28)."""
    if len(alg) != 2:
        raise ValueError(f"Invalid algebraic coordinate: '{alg}'")
    file_char, rank_char = alg[0].lower(), alg[1]
    if file_char not in "abcdefgh" or rank_char not in "12345678":
        raise ValueError(f"Invalid algebraic coordinate: '{alg}'")
    file = ord(file_char) - ord("a")
    rank = int(rank_char) - 1
    return coords_to_square(file, rank)
