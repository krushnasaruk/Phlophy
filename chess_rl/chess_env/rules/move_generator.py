"""
Deterministic pure-Python chess move generator and rules engine.
Implements strict separation between pseudo-legal and legal move validation.
"""

from typing import List, Optional, Tuple, Dict
from chess_rl.chess_env.board.types import (
    Color,
    PieceType,
    Piece,
    CastlingRights,
    square_to_coords,
    coords_to_square,
    E1, G1, C1, H1, A1, F1, D1, B1,
    E8, G8, C8, H8, A8, F8, D8, B8,
)
from chess_rl.chess_env.board.move import Move
from chess_rl.chess_env.rules.game_state import GameState


# Precomputed knight move offsets (df, dr)
KNIGHT_OFFSETS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]

# Precomputed king move offsets (df, dr)
KING_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Precomputed sliding rays (df, dr)
BISHOP_RAYS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ROOK_RAYS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
QUEEN_RAYS = BISHOP_RAYS + ROOK_RAYS


class MoveGenerator:
    """
    Complete FIDE chess rules engine and legal move generator.
    """

    @classmethod
    def find_king(cls, color: Color, state: GameState) -> Optional[int]:
        """Locate square index of the King of the specified color."""
        for sq in range(64):
            p = state.board[sq]
            if p is not None and p.piece_type == PieceType.KING and p.color == color:
                return sq
        return None

    @classmethod
    def is_square_attacked(cls, square: int, attacking_color: Color, state: GameState) -> bool:
        """
        Determine whether 'square' is under attack by any piece of 'attacking_color'.
        """
        sq_file, sq_rank = square_to_coords(square)

        # 1. Pawn attacks
        # If attacking color is White, White pawns attack diagonally UP (+1 rank),
        # meaning attacking White pawns would be on sq_rank - 1.
        pawn_rank_delta = 1 if attacking_color == Color.WHITE else -1
        pawn_source_rank = sq_rank - pawn_rank_delta
        if 0 <= pawn_source_rank < 8:
            for d_file in [-1, 1]:
                pawn_source_file = sq_file + d_file
                if 0 <= pawn_source_file < 8:
                    src_sq = coords_to_square(pawn_source_file, pawn_source_rank)
                    p = state.board[src_sq]
                    if p is not None and p.color == attacking_color and p.piece_type == PieceType.PAWN:
                        return True

        # 2. Knight attacks
        for df, dr in KNIGHT_OFFSETS:
            kf, kr = sq_file + df, sq_rank + dr
            if 0 <= kf < 8 and 0 <= kr < 8:
                src_sq = coords_to_square(kf, kr)
                p = state.board[src_sq]
                if p is not None and p.color == attacking_color and p.piece_type == PieceType.KNIGHT:
                    return True

        # 3. Bishop and Queen (Diagonal rays)
        for df, dr in BISHOP_RAYS:
            f, r = sq_file + df, sq_rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                src_sq = coords_to_square(f, r)
                p = state.board[src_sq]
                if p is not None:
                    if p.color == attacking_color and p.piece_type in (PieceType.BISHOP, PieceType.QUEEN):
                        return True
                    break  # Ray is blocked
                f += df
                r += dr

        # 4. Rook and Queen (Orthogonal rays)
        for df, dr in ROOK_RAYS:
            f, r = sq_file + df, sq_rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                src_sq = coords_to_square(f, r)
                p = state.board[src_sq]
                if p is not None:
                    if p.color == attacking_color and p.piece_type in (PieceType.ROOK, PieceType.QUEEN):
                        return True
                    break  # Ray is blocked
                f += df
                r += dr

        # 5. King attacks (adjacent squares)
        for df, dr in KING_OFFSETS:
            kf, kr = sq_file + df, sq_rank + dr
            if 0 <= kf < 8 and 0 <= kr < 8:
                src_sq = coords_to_square(kf, kr)
                p = state.board[src_sq]
                if p is not None and p.color == attacking_color and p.piece_type == PieceType.KING:
                    return True

        return False

    @classmethod
    def is_in_check(cls, color: Color, state: GameState) -> bool:
        """Check if the King of the specified color is in check."""
        king_sq = cls.find_king(color, state)
        if king_sq is None:
            return False
        return cls.is_square_attacked(king_sq, color.opponent, state)

    @classmethod
    def generate_pseudo_legal_moves(cls, state: GameState) -> List[Move]:
        """
        Generate all pseudo-legal moves for the active side to move
        (does not check if own king is left in check).
        """
        moves: List[Move] = []
        color = state.side_to_move
        pawn_dir = 1 if color == Color.WHITE else -1
        start_pawn_rank = 1 if color == Color.WHITE else 6
        promo_rank = 7 if color == Color.WHITE else 0

        for sq in range(64):
            piece = state.board[sq]
            if piece is None or piece.color != color:
                continue

            file, rank = square_to_coords(sq)

            # PAWN MOVEMENT
            if piece.piece_type == PieceType.PAWN:
                # 1. Single forward push
                fwd_rank = rank + pawn_dir
                if 0 <= fwd_rank < 8:
                    fwd_sq = coords_to_square(file, fwd_rank)
                    if state.board[fwd_sq] is None:
                        if fwd_rank == promo_rank:
                            for promo in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
                                moves.append(Move(from_square=sq, to_square=fwd_sq, promotion_piece=promo))
                        else:
                            moves.append(Move(from_square=sq, to_square=fwd_sq))

                        # 2. Double forward push from start rank
                        if rank == start_pawn_rank:
                            double_fwd_rank = rank + 2 * pawn_dir
                            double_fwd_sq = coords_to_square(file, double_fwd_rank)
                            if state.board[double_fwd_sq] is None:
                                moves.append(Move(from_square=sq, to_square=double_fwd_sq))

                # 3. Diagonal Captures
                for df in [-1, 1]:
                    target_file = file + df
                    if 0 <= target_file < 8 and 0 <= fwd_rank < 8:
                        cap_sq = coords_to_square(target_file, fwd_rank)
                        target_piece = state.board[cap_sq]
                        
                        # Standard capture
                        if target_piece is not None and target_piece.color == color.opponent:
                            if fwd_rank == promo_rank:
                                for promo in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
                                    moves.append(Move(from_square=sq, to_square=cap_sq, promotion_piece=promo))
                            else:
                                moves.append(Move(from_square=sq, to_square=cap_sq))

                        # En passant capture
                        elif cap_sq == state.en_passant_target:
                            moves.append(Move(from_square=sq, to_square=cap_sq, is_en_passant=True))

            # KNIGHT MOVEMENT
            elif piece.piece_type == PieceType.KNIGHT:
                for df, dr in KNIGHT_OFFSETS:
                    tf, tr = file + df, rank + dr
                    if 0 <= tf < 8 and 0 <= tr < 8:
                        target_sq = coords_to_square(tf, tr)
                        target_p = state.board[target_sq]
                        if target_p is None or target_p.color == color.opponent:
                            moves.append(Move(from_square=sq, to_square=target_sq))

            # BISHOP MOVEMENT
            elif piece.piece_type == PieceType.BISHOP:
                for df, dr in BISHOP_RAYS:
                    f, r = file + df, rank + dr
                    while 0 <= f < 8 and 0 <= r < 8:
                        target_sq = coords_to_square(f, r)
                        target_p = state.board[target_sq]
                        if target_p is None:
                            moves.append(Move(from_square=sq, to_square=target_sq))
                        else:
                            if target_p.color == color.opponent:
                                moves.append(Move(from_square=sq, to_square=target_sq))
                            break
                        f += df
                        r += dr

            # ROOK MOVEMENT
            elif piece.piece_type == PieceType.ROOK:
                for df, dr in ROOK_RAYS:
                    f, r = file + df, rank + dr
                    while 0 <= f < 8 and 0 <= r < 8:
                        target_sq = coords_to_square(f, r)
                        target_p = state.board[target_sq]
                        if target_p is None:
                            moves.append(Move(from_square=sq, to_square=target_sq))
                        else:
                            if target_p.color == color.opponent:
                                moves.append(Move(from_square=sq, to_square=target_sq))
                            break
                        f += df
                        r += dr

            # QUEEN MOVEMENT
            elif piece.piece_type == PieceType.QUEEN:
                for df, dr in QUEEN_RAYS:
                    f, r = file + df, rank + dr
                    while 0 <= f < 8 and 0 <= r < 8:
                        target_sq = coords_to_square(f, r)
                        target_p = state.board[target_sq]
                        if target_p is None:
                            moves.append(Move(from_square=sq, to_square=target_sq))
                        else:
                            if target_p.color == color.opponent:
                                moves.append(Move(from_square=sq, to_square=target_sq))
                            break
                        f += df
                        r += dr

            # KING MOVEMENT (normal steps)
            elif piece.piece_type == PieceType.KING:
                for df, dr in KING_OFFSETS:
                    tf, tr = file + df, rank + dr
                    if 0 <= tf < 8 and 0 <= tr < 8:
                        target_sq = coords_to_square(tf, tr)
                        target_p = state.board[target_sq]
                        if target_p is None or target_p.color == color.opponent:
                            moves.append(Move(from_square=sq, to_square=target_sq))

        return moves

    @classmethod
    def _generate_castling_moves(cls, state: GameState) -> List[Move]:
        """Generate legal castling moves for the active player."""
        moves: List[Move] = []
        color = state.side_to_move
        opp_color = color.opponent

        if color == Color.WHITE:
            # White Kingside: King e1 (4) -> g1 (6)
            if state.castling_rights.white_kingside:
                # 1. Check rook presence on h1 (7)
                rook = state.board[H1]
                if rook is not None and rook.piece_type == PieceType.ROOK and rook.color == Color.WHITE:
                    # 2. Check squares between e1 and h1 are empty: f1 (5), g1 (6)
                    if state.board[F1] is None and state.board[G1] is None:
                        # 3. King not currently in check, and does not pass through/land on attacked squares
                        if (
                            not cls.is_square_attacked(E1, opp_color, state)
                            and not cls.is_square_attacked(F1, opp_color, state)
                            and not cls.is_square_attacked(G1, opp_color, state)
                        ):
                            moves.append(Move(from_square=E1, to_square=G1, is_castling=True))

            # White Queenside: King e1 (4) -> c1 (2)
            if state.castling_rights.white_queenside:
                # 1. Check rook presence on a1 (0)
                rook = state.board[A1]
                if rook is not None and rook.piece_type == PieceType.ROOK and rook.color == Color.WHITE:
                    # 2. Check squares between e1 and a1 are empty: b1 (1), c1 (2), d1 (3)
                    if state.board[B1] is None and state.board[C1] is None and state.board[D1] is None:
                        # 3. King not in check, does not pass through d1, does not land on c1
                        if (
                            not cls.is_square_attacked(E1, opp_color, state)
                            and not cls.is_square_attacked(D1, opp_color, state)
                            and not cls.is_square_attacked(C1, opp_color, state)
                        ):
                            moves.append(Move(from_square=E1, to_square=C1, is_castling=True))

        else:  # BLACK
            # Black Kingside: King e8 (60) -> g8 (62)
            if state.castling_rights.black_kingside:
                rook = state.board[H8]
                if rook is not None and rook.piece_type == PieceType.ROOK and rook.color == Color.BLACK:
                    if state.board[F8] is None and state.board[G8] is None:
                        if (
                            not cls.is_square_attacked(E8, opp_color, state)
                            and not cls.is_square_attacked(F8, opp_color, state)
                            and not cls.is_square_attacked(G8, opp_color, state)
                        ):
                            moves.append(Move(from_square=E8, to_square=G8, is_castling=True))

            # Black Queenside: King e8 (60) -> c8 (58)
            if state.castling_rights.black_queenside:
                rook = state.board[A8]
                if rook is not None and rook.piece_type == PieceType.ROOK and rook.color == Color.BLACK:
                    if state.board[B8] is None and state.board[C8] is None and state.board[D8] is None:
                        if (
                            not cls.is_square_attacked(E8, opp_color, state)
                            and not cls.is_square_attacked(D8, opp_color, state)
                            and not cls.is_square_attacked(C8, opp_color, state)
                        ):
                            moves.append(Move(from_square=E8, to_square=C8, is_castling=True))

        return moves

    @classmethod
    def generate_legal_moves(cls, state: GameState) -> List[Move]:
        """
        Generate all strictly legal moves for the active player.
        Filters pseudo-legal moves by ensuring the player's King is not left in check,
        and adds verified legal castling moves.
        """
        legal_moves: List[Move] = []
        pseudo_moves = cls.generate_pseudo_legal_moves(state)

        # Filter pseudo-legal moves
        for move in pseudo_moves:
            next_state = cls.apply_move(state, move)
            # Check if moving player's king is left in check
            if not cls.is_in_check(state.side_to_move, next_state):
                legal_moves.append(move)

        # Castling moves are independently verified for king safety
        castling_moves = cls._generate_castling_moves(state)
        legal_moves.extend(castling_moves)

        return legal_moves

    @classmethod
    def apply_move(cls, state: GameState, move: Move) -> GameState:
        """
        Execute a move on the given GameState and return the new resulting GameState.
        Pure and non-destructive.
        """
        new_state = state.copy()
        moving_piece = new_state.board[move.from_square]
        target_piece = new_state.board[move.to_square]
        color = state.side_to_move

        if moving_piece is None:
            raise ValueError(f"No piece at source square {move.from_square} for move {move}")

        # 1. Update Board Position
        new_state.board[move.from_square] = None

        if move.is_castling:
            # Move King and corresponding Rook
            if color == Color.WHITE:
                if move.to_square == G1:  # Kingside
                    new_state.board[G1] = moving_piece
                    new_state.board[F1] = new_state.board[H1]
                    new_state.board[H1] = None
                elif move.to_square == C1:  # Queenside
                    new_state.board[C1] = moving_piece
                    new_state.board[D1] = new_state.board[A1]
                    new_state.board[A1] = None
            else:  # BLACK
                if move.to_square == G8:  # Kingside
                    new_state.board[G8] = moving_piece
                    new_state.board[F8] = new_state.board[H8]
                    new_state.board[H8] = None
                elif move.to_square == C8:  # Queenside
                    new_state.board[C8] = moving_piece
                    new_state.board[D8] = new_state.board[A8]
                    new_state.board[A8] = None

        elif move.is_en_passant:
            # Attacking pawn moves to en_passant_target, captured pawn is removed
            new_state.board[move.to_square] = moving_piece
            from_file, from_rank = square_to_coords(move.from_square)
            to_file, _ = square_to_coords(move.to_square)
            captured_sq = coords_to_square(to_file, from_rank)
            new_state.board[captured_sq] = None

        elif move.promotion_piece is not None:
            new_state.board[move.to_square] = Piece(move.promotion_piece, color)

        else:
            new_state.board[move.to_square] = moving_piece

        # 2. Update En Passant Target
        new_state.en_passant_target = None
        if moving_piece.piece_type == PieceType.PAWN:
            from_file, from_rank = square_to_coords(move.from_square)
            _, to_rank = square_to_coords(move.to_square)
            if abs(to_rank - from_rank) == 2:
                # Set en passant target square between from and to
                ep_rank = (from_rank + to_rank) // 2
                new_state.en_passant_target = coords_to_square(from_file, ep_rank)

        # 3. Update Castling Rights
        # If King moves
        if moving_piece.piece_type == PieceType.KING:
            if color == Color.WHITE:
                new_state.castling_rights.white_kingside = False
                new_state.castling_rights.white_queenside = False
            else:
                new_state.castling_rights.black_kingside = False
                new_state.castling_rights.black_queenside = False

        # If Rook moves
        if moving_piece.piece_type == PieceType.ROOK:
            if move.from_square == H1:
                new_state.castling_rights.white_kingside = False
            elif move.from_square == A1:
                new_state.castling_rights.white_queenside = False
            elif move.from_square == H8:
                new_state.castling_rights.black_kingside = False
            elif move.from_square == A8:
                new_state.castling_rights.black_queenside = False

        # If Rook is captured on its corner
        if target_piece is not None and target_piece.piece_type == PieceType.ROOK:
            if move.to_square == H1:
                new_state.castling_rights.white_kingside = False
            elif move.to_square == A1:
                new_state.castling_rights.white_queenside = False
            elif move.to_square == H8:
                new_state.castling_rights.black_kingside = False
            elif move.to_square == A8:
                new_state.castling_rights.black_queenside = False

        # 4. Update Halfmove Clock (50-move rule counter)
        is_capture = (target_piece is not None) or move.is_en_passant
        is_pawn_move = moving_piece.piece_type == PieceType.PAWN
        if is_capture or is_pawn_move:
            new_state.halfmove_clock = 0
        else:
            new_state.halfmove_clock += 1

        # 5. Update Fullmove Number & Side to Move
        if color == Color.BLACK:
            new_state.fullmove_number += 1
        new_state.side_to_move = color.opponent

        # 6. Record Position Key in History
        key = new_state.position_key()
        new_state.position_history[key] = new_state.position_history.get(key, 0) + 1

        return new_state

    @classmethod
    def is_checkmate(cls, state: GameState) -> bool:
        """Check if current position is checkmate."""
        return cls.is_in_check(state.side_to_move, state) and len(cls.generate_legal_moves(state)) == 0

    @classmethod
    def is_stalemate(cls, state: GameState) -> bool:
        """Check if current position is stalemate."""
        return not cls.is_in_check(state.side_to_move, state) and len(cls.generate_legal_moves(state)) == 0

    @classmethod
    def is_fifty_move_draw(cls, state: GameState) -> bool:
        """Check if 50-move rule applies (100 halfmoves without capture or pawn move)."""
        return state.halfmove_clock >= 100

    @classmethod
    def is_threefold_repetition(cls, state: GameState) -> bool:
        """Check if current position has occurred 3 or more times."""
        return state.position_history.get(state.position_key(), 0) >= 3

    @classmethod
    def is_insufficient_material(cls, state: GameState) -> bool:
        """
        Check for draw by insufficient material:
        - King vs King
        - King + Bishop vs King
        - King + Knight vs King
        - King + Bishop vs King + Bishop (if same color square bishops)
        """
        white_pieces = []
        black_pieces = []
        bishop_squares = []

        for sq in range(64):
            p = state.board[sq]
            if p is not None:
                if p.piece_type != PieceType.KING:
                    if p.color == Color.WHITE:
                        white_pieces.append(p.piece_type)
                    else:
                        black_pieces.append(p.piece_type)
                    if p.piece_type == PieceType.BISHOP:
                        f, r = square_to_coords(sq)
                        bishop_squares.append((f + r) % 2)

        # K vs K
        if len(white_pieces) == 0 and len(black_pieces) == 0:
            return True

        # K+N vs K or K+B vs K
        if (len(white_pieces) == 1 and white_pieces[0] in (PieceType.KNIGHT, PieceType.BISHOP) and len(black_pieces) == 0) or (
            len(black_pieces) == 1 and black_pieces[0] in (PieceType.KNIGHT, PieceType.BISHOP) and len(white_pieces) == 0
        ):
            return True

        # K+B vs K+B (same color bishops)
        if len(white_pieces) == 1 and white_pieces[0] == PieceType.BISHOP and len(black_pieces) == 1 and black_pieces[0] == PieceType.BISHOP:
            if len(bishop_squares) == 2 and bishop_squares[0] == bishop_squares[1]:
                return True

        return False

    @classmethod
    def is_game_over(cls, state: GameState) -> bool:
        """Check if the game has ended by any rule."""
        if cls.is_checkmate(state) or cls.is_stalemate(state):
            return True
        if cls.is_fifty_move_draw(state) or cls.is_threefold_repetition(state):
            return True
        if cls.is_insufficient_material(state):
            return True
        return False
