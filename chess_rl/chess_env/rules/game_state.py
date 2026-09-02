"""
GameState representation with complete board tracking, FEN serialization, and ASCII rendering.
"""

from typing import List, Optional, Dict, Tuple
from chess_rl.chess_env.board.types import (
    Color,
    PieceType,
    Piece,
    CastlingRights,
    square_to_algebraic,
    algebraic_to_square,
    coords_to_square,
)


class GameState:
    """
    Encapsulates a full, discrete chess game state.
    """

    def __init__(
        self,
        board: Optional[List[Optional[Piece]]] = None,
        side_to_move: Color = Color.WHITE,
        castling_rights: Optional[CastlingRights] = None,
        en_passant_target: Optional[int] = None,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
        position_history: Optional[Dict[str, int]] = None,
    ):
        if board is None:
            self.board: List[Optional[Piece]] = [None] * 64
        else:
            if len(board) != 64:
                raise ValueError("Board must have exactly 64 squares.")
            self.board = list(board)

        self.side_to_move: Color = side_to_move
        self.castling_rights: CastlingRights = (
            castling_rights.copy() if castling_rights else CastlingRights()
        )
        self.en_passant_target: Optional[int] = en_passant_target
        self.halfmove_clock: int = halfmove_clock
        self.fullmove_number: int = fullmove_number
        self.position_history: Dict[str, int] = (
            dict(position_history) if position_history is not None else {}
        )

    def get_piece(self, square: int) -> Optional[Piece]:
        """Return piece at square index [0..63] or None."""
        return self.board[square]

    def set_piece(self, square: int, piece: Optional[Piece]) -> None:
        """Place piece on square index [0..63]."""
        self.board[square] = piece

    def copy(self) -> "GameState":
        """Return an independent deep copy of the game state."""
        return GameState(
            board=self.board,
            side_to_move=self.side_to_move,
            castling_rights=self.castling_rights,
            en_passant_target=self.en_passant_target,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
            position_history=self.position_history,
        )

    def position_key(self) -> str:
        """
        Generate a unique state signature for threefold repetition detection:
        Includes piece placement, side to move, castling rights, and en passant target.
        """
        # Board placement string
        piece_str = "".join(p.symbol if p else "." for p in self.board)
        side_str = "w" if self.side_to_move == Color.WHITE else "b"
        castling_str = self.castling_rights.to_fen()
        ep_str = (
            square_to_algebraic(self.en_passant_target)
            if self.en_passant_target is not None
            else "-"
        )
        return f"{piece_str} {side_str} {castling_str} {ep_str}"

    @classmethod
    def initial(cls) -> "GameState":
        """Construct the standard chess starting position."""
        state = cls()
        # White major pieces (Rank 1: squares 0..7)
        state.board[0] = Piece(PieceType.ROOK, Color.WHITE)
        state.board[1] = Piece(PieceType.KNIGHT, Color.WHITE)
        state.board[2] = Piece(PieceType.BISHOP, Color.WHITE)
        state.board[3] = Piece(PieceType.QUEEN, Color.WHITE)
        state.board[4] = Piece(PieceType.KING, Color.WHITE)
        state.board[5] = Piece(PieceType.BISHOP, Color.WHITE)
        state.board[6] = Piece(PieceType.KNIGHT, Color.WHITE)
        state.board[7] = Piece(PieceType.ROOK, Color.WHITE)

        # White pawns (Rank 2: squares 8..15)
        for sq in range(8, 16):
            state.board[sq] = Piece(PieceType.PAWN, Color.WHITE)

        # Black pawns (Rank 7: squares 48..55)
        for sq in range(48, 56):
            state.board[sq] = Piece(PieceType.PAWN, Color.BLACK)

        # Black major pieces (Rank 8: squares 56..63)
        state.board[56] = Piece(PieceType.ROOK, Color.BLACK)
        state.board[57] = Piece(PieceType.KNIGHT, Color.BLACK)
        state.board[58] = Piece(PieceType.BISHOP, Color.BLACK)
        state.board[59] = Piece(PieceType.QUEEN, Color.BLACK)
        state.board[60] = Piece(PieceType.KING, Color.BLACK)
        state.board[61] = Piece(PieceType.BISHOP, Color.BLACK)
        state.board[62] = Piece(PieceType.KNIGHT, Color.BLACK)
        state.board[63] = Piece(PieceType.ROOK, Color.BLACK)

        state.side_to_move = Color.WHITE
        state.castling_rights = CastlingRights(True, True, True, True)
        state.en_passant_target = None
        state.halfmove_clock = 0
        state.fullmove_number = 1
        state.position_history = {state.position_key(): 1}
        return state

    @classmethod
    def from_fen(cls, fen: str) -> "GameState":
        """
        Parse standard Forsyth-Edwards Notation (FEN) into GameState.
        Example: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        """
        parts = fen.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Invalid FEN string (insufficient fields): '{fen}'")

        placement, side_str, castling_str, ep_str = parts[:4]
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1

        board: List[Optional[Piece]] = [None] * 64
        ranks = placement.split("/")
        if len(ranks) != 8:
            raise ValueError(f"Invalid FEN board placement (expected 8 ranks): '{placement}'")

        # FEN starts from Rank 8 (top) down to Rank 1 (bottom)
        for rank_idx, rank_str in enumerate(ranks):
            actual_rank = 7 - rank_idx
            file_idx = 0
            for char in rank_str:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    piece_type, color = PieceType.from_symbol(char)
                    sq = coords_to_square(file_idx, actual_rank)
                    board[sq] = Piece(piece_type, color)
                    file_idx += 1
            if file_idx != 8:
                raise ValueError(f"Invalid FEN rank '{rank_str}' in rank {actual_rank + 1}")

        side_to_move = Color.WHITE if side_str.lower() == "w" else Color.BLACK
        castling_rights = CastlingRights.from_fen(castling_str)
        ep_target = algebraic_to_square(ep_str) if ep_str != "-" else None

        state = cls(
            board=board,
            side_to_move=side_to_move,
            castling_rights=castling_rights,
            en_passant_target=ep_target,
            halfmove_clock=halfmove,
            fullmove_number=fullmove,
        )
        state.position_history = {state.position_key(): 1}
        return state

    def to_fen(self) -> str:
        """Serialize GameState to standard FEN string."""
        ranks_fen = []
        for rank in range(7, -1, -1):
            rank_str = ""
            empty_count = 0
            for file in range(8):
                sq = coords_to_square(file, rank)
                piece = self.board[sq]
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += piece.symbol
            if empty_count > 0:
                rank_str += str(empty_count)
            ranks_fen.append(rank_str)

        placement_str = "/".join(ranks_fen)
        side_str = "w" if self.side_to_move == Color.WHITE else "b"
        castling_str = self.castling_rights.to_fen()
        ep_str = (
            square_to_algebraic(self.en_passant_target)
            if self.en_passant_target is not None
            else "-"
        )

        return f"{placement_str} {side_str} {castling_str} {ep_str} {self.halfmove_clock} {self.fullmove_number}"

    def render_ascii(self) -> str:
        """Render board in human-readable ASCII format."""
        lines = []
        lines.append("  +-----------------+")
        for rank in range(7, -1, -1):
            row_pieces = []
            for file in range(8):
                sq = coords_to_square(file, rank)
                p = self.board[sq]
                row_pieces.append(p.symbol if p else ".")
            lines.append(f"{rank + 1} | {' '.join(row_pieces)} |")
        lines.append("  +-----------------+")
        lines.append("    a b c d e f g h")
        lines.append(f"Side to move: {self.side_to_move}")
        lines.append(f"Castling: {self.castling_rights.to_fen()} | EP: {square_to_algebraic(self.en_passant_target) if self.en_passant_target is not None else '-'}")
        lines.append(f"Halfmove clock: {self.halfmove_clock} | Fullmove: {self.fullmove_number}")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return (
            self.board == other.board
            and self.side_to_move == other.side_to_move
            and self.castling_rights == other.castling_rights
            and self.en_passant_target == other.en_passant_target
            and self.halfmove_clock == other.halfmove_clock
            and self.fullmove_number == other.fullmove_number
        )
