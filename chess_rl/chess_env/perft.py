"""
PERFT (Performance Test / Move Path Enumerator) validation engine.
Calculates legal move tree leaf node counts at arbitrary depths.
"""

import time
from typing import Dict, List, Tuple
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator


def perft(state: GameState, depth: int) -> int:
    """
    Count the number of leaf nodes in the legal move tree at given depth.
    """
    if depth == 0:
        return 1

    moves = MoveGenerator.generate_legal_moves(state)
    if depth == 1:
        return len(moves)

    nodes = 0
    for move in moves:
        next_state = MoveGenerator.apply_move(state, move)
        nodes += perft(next_state, depth - 1)

    return nodes


def perft_divide(state: GameState, depth: int) -> Dict[str, int]:
    """
    Compute PERFT divide: node count for each individual root legal move.
    """
    if depth <= 0:
        return {}

    moves = MoveGenerator.generate_legal_moves(state)
    divide_counts = {}

    for move in moves:
        next_state = MoveGenerator.apply_move(state, move)
        nodes = perft(next_state, depth - 1)
        divide_counts[move.to_uci()] = nodes

    return divide_counts


# Standard benchmark test positions with known canonical node counts
PERFT_TEST_SUITE = [
    {
        "name": "Position 1 - Initial Position",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "expected": {1: 20, 2: 400, 3: 8902, 4: 197281},
    },
    {
        "name": "Position 2 - Kiwipete (Complex Castling & Pins)",
        "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "expected": {1: 48, 2: 2039, 3: 97862},
    },
    {
        "name": "Position 3 - Endgame & Pins",
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "expected": {1: 14, 2: 191, 3: 2812},
    },
    {
        "name": "Position 4 - Promotions & Checks",
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "expected": {1: 6, 2: 264, 3: 9467},
    },
    {
        "name": "Position 5 - Promotions & Check Discoveries",
        "fen": "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        "expected": {1: 24, 2: 496},
    },
]


def run_standard_perft_suite(max_initial_depth: int = 4) -> bool:
    """
    Run the full standard PERFT suite and print execution diagnostics.
    """
    print("=" * 70)
    print("RUNNING CHESS RULES ENGINE PERFT VALIDATION SUITE")
    print("=" * 70)

    all_passed = True
    for item in PERFT_TEST_SUITE:
        name = item["name"]
        fen = item["fen"]
        expected_dict = item["expected"]

        print(f"\n[TEST POSITION] {name}")
        print(f"  FEN: {fen}")
        state = GameState.from_fen(fen)

        for depth, expected_nodes in expected_dict.items():
            if name.startswith("Position 1") and depth > max_initial_depth:
                continue
            
            t0 = time.perf_counter()
            actual_nodes = perft(state, depth)
            elapsed = time.perf_counter() - t0
            nps = int(actual_nodes / elapsed) if elapsed > 0 else 0

            passed = (actual_nodes == expected_nodes)
            status = "PASSED" if passed else "FAILED"
            print(
                f"  Depth {depth}: Nodes = {actual_nodes:,} "
                f"(Expected: {expected_nodes:,}) | Time: {elapsed:.3f}s ({nps:,} nps) -> [{status}]"
            )

            if not passed:
                all_passed = False
                print("  --> PERFT DIVIDE MISMATCH DEBUG:")
                div = perft_divide(state, depth)
                for move_str, count in sorted(div.items()):
                    print(f"      {move_str}: {count}")

    print("\n" + "=" * 70)
    if all_passed:
        print("PERFT VALIDATION: ALL POSITIONS PASSED PERFECTLY")
    else:
        print("PERFT VALIDATION: ONE OR MORE POSITIONS FAILED")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    run_standard_perft_suite(max_initial_depth=4)
