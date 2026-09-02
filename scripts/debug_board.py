"""
Debug Board CLI: Development utility for inspecting positions, validating moves, and testing PERFT.
"""

import argparse
import sys
from pathlib import Path

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator
from chess_rl.chess_env.board.move import Move
from chess_rl.chess_env.perft import perft, perft_divide, run_standard_perft_suite


def parse_args():
    parser = argparse.ArgumentParser(description="Debug and inspect chess positions.")
    parser.add_argument(
        "--fen",
        "-f",
        type=str,
        default=None,
        help="FEN string to load (defaults to starting position)",
    )
    parser.add_argument(
        "--move",
        "-m",
        type=str,
        default=None,
        help="Apply a move in UCI notation (e.g. 'e2e4')",
    )
    parser.add_argument(
        "--perft",
        "-p",
        type=int,
        default=None,
        help="Run PERFT node counting to specified depth",
    )
    parser.add_argument(
        "--perft-suite",
        action="store_true",
        help="Run full standard PERFT test suite",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.perft_suite:
        success = run_standard_perft_suite()
        return 0 if success else 1

    # Load state
    if args.fen:
        state = GameState.from_fen(args.fen)
    else:
        state = GameState.initial()

    # Apply move if specified
    if args.move:
        legal_moves = MoveGenerator.generate_legal_moves(state)
        matching_move = None
        for m in legal_moves:
            if m.to_uci() == args.move:
                matching_move = m
                break
        if matching_move is None:
            print(f"Error: Move '{args.move}' is illegal in current position!")
            print(f"Legal moves: {[m.to_uci() for m in legal_moves]}")
            return 1
        state = MoveGenerator.apply_move(state, matching_move)
        print(f"Applied move: {matching_move.to_uci()}\n")

    # Render board
    print(state.render_ascii())
    print(f"FEN: {state.to_fen()}")

    # Legal moves & status
    in_check = MoveGenerator.is_in_check(state.side_to_move, state)
    legal_moves = MoveGenerator.generate_legal_moves(state)
    is_mate = MoveGenerator.is_checkmate(state)
    is_stale = MoveGenerator.is_stalemate(state)

    print(f"\nStatus:")
    print(f"  In Check: {in_check}")
    print(f"  Checkmate: {is_mate}")
    print(f"  Stalemate: {is_stale}")
    print(f"  Legal Move Count: {len(legal_moves)}")
    print(f"  Legal Moves: {', '.join(sorted([m.to_uci() for m in legal_moves]))}")

    # Run PERFT if requested
    if args.perft is not None:
        print(f"\n--- Running PERFT (Depth {args.perft}) ---")
        divide = perft_divide(state, args.perft)
        total_nodes = sum(divide.values())
        for move_str, count in sorted(divide.items()):
            print(f"  {move_str}: {count:,}")
        print(f"Total Nodes: {total_nodes:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
