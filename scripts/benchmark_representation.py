"""
Representation and rules engine performance benchmarking script.
Measures execution latencies for observation encoding, move generation, action encoding, decoding, and masking.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List
import numpy as np

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_rl.chess_env.environments.standard_chess import StandardChessEnvironment
from chess_rl.chess_env.board.action_space import ActionEncoder
from chess_rl.chess_env.rules.game_state import GameState
from chess_rl.chess_env.rules.move_generator import MoveGenerator
from chess_rl.utils.system_info import collect_system_info
from chess_rl.utils.seeding import seed_everything
from chess_rl.config.schema import ExperimentConfig, ExperimentMetaConfig
from chess_rl.tracking.experiment_tracker import ExperimentTracker


def generate_benchmark_positions(num_positions: int = 500, seed: int = 42) -> List[GameState]:
    """Generate diverse valid legal chess positions via random rollouts."""
    seed_everything(seed)
    rng = random.Random(seed)
    positions = []

    while len(positions) < num_positions:
        env = StandardChessEnvironment()
        res = env.reset()
        positions.append(env.state.copy())

        # Play random game up to 60 moves, sampling states
        for _ in range(60):
            legal_moves = env.legal_actions()
            if not legal_moves or env.is_terminal():
                break
            move = rng.choice(legal_moves)
            env.step(move)
            positions.append(env.state.copy())
            if len(positions) >= num_positions:
                break

    return positions[:num_positions]


def run_benchmark(num_positions: int = 500, seed: int = 42, output_dir: str = "experiments/results"):
    print("=" * 75)
    print("CHESS REPRESENTATION & RULES LATENCY BENCHMARK")
    print("=" * 75)
    print(f"Generating {num_positions} legal test positions (seed={seed})...")

    positions = generate_benchmark_positions(num_positions=num_positions, seed=seed)
    env = StandardChessEnvironment()

    print(f"Collected {len(positions)} valid positions. Commencing microsecond timing...")

    # 1. Benchmark Observation Encoding
    t0 = time.perf_counter()
    for state in positions:
        env.state = state
        _ = env.get_observation_tensor()
    obs_elapsed = time.perf_counter() - t0
    obs_us = (obs_elapsed / len(positions)) * 1e6
    obs_thru = int(len(positions) / obs_elapsed) if obs_elapsed > 0 else 0

    # 2. Benchmark Legal Move Generation
    t0 = time.perf_counter()
    all_legal_moves = []
    for state in positions:
        moves = MoveGenerator.generate_legal_moves(state)
        all_legal_moves.append(moves)
    gen_elapsed = time.perf_counter() - t0
    gen_us = (gen_elapsed / len(positions)) * 1e6
    gen_thru = int(len(positions) / gen_elapsed) if gen_elapsed > 0 else 0

    # 3. Benchmark Action Encoding
    flat_moves = [m for moves in all_legal_moves for m in moves]
    t0 = time.perf_counter()
    encoded_ids = []
    for m in flat_moves:
        encoded_ids.append(ActionEncoder.encode(m))
    enc_elapsed = time.perf_counter() - t0
    enc_us = (enc_elapsed / len(flat_moves)) * 1e6 if flat_moves else 0
    enc_thru = int(len(flat_moves) / enc_elapsed) if enc_elapsed > 0 else 0

    # 4. Benchmark Action Decoding
    t0 = time.perf_counter()
    for idx in encoded_ids:
        _ = ActionEncoder.decode(idx)
    dec_elapsed = time.perf_counter() - t0
    dec_us = (dec_elapsed / len(encoded_ids)) * 1e6 if encoded_ids else 0
    dec_thru = int(len(encoded_ids) / dec_elapsed) if dec_elapsed > 0 else 0

    # 5. Benchmark Legal Mask Generation
    t0 = time.perf_counter()
    for moves in all_legal_moves:
        _ = ActionEncoder.create_legal_mask(moves)
    mask_elapsed = time.perf_counter() - t0
    mask_us = (mask_elapsed / len(all_legal_moves)) * 1e6
    mask_thru = int(len(all_legal_moves) / mask_elapsed) if mask_elapsed > 0 else 0

    # Print Results Table
    print("\n" + "-" * 75)
    print(f"{'Operation':<35} | {'Latency (us/op)':<18} | {'Throughput (ops/sec)':<18}")
    print("-" * 75)
    print(f"{'1. Observation Tensor Encoding':<35} | {obs_us:>14.2f} us | {obs_thru:>15,} ops/s")
    print(f"{'2. Legal Move Generation':<35} | {gen_us:>14.2f} us | {gen_thru:>15,} ops/s")
    print(f"{'3. Action ID Encoding':<35} | {enc_us:>14.2f} us | {enc_thru:>15,} ops/s")
    print(f"{'4. Action ID Decoding':<35} | {dec_us:>14.2f} us | {dec_thru:>15,} ops/s")
    print(f"{'5. Legal Action Mask Construction':<35} | {mask_us:>14.2f} us | {mask_thru:>15,} ops/s")
    print("-" * 75)

    # Save to Experiment Results Directory
    metrics_payload = {
        "num_positions": len(positions),
        "total_moves_tested": len(flat_moves),
        "seed": seed,
        "latencies_us": {
            "observation_encoding": obs_us,
            "legal_move_generation": gen_us,
            "action_encoding": enc_us,
            "action_decoding": dec_us,
            "legal_mask_generation": mask_us,
        },
        "throughput_ops_per_sec": {
            "observation_encoding": obs_thru,
            "legal_move_generation": gen_thru,
            "action_encoding": enc_thru,
            "action_decoding": dec_thru,
            "legal_mask_generation": mask_thru,
        },
    }

    # Initialize tracker
    cfg = ExperimentConfig(
        experiment=ExperimentMetaConfig(
            name="representation_benchmark",
            description="Microsecond performance benchmark of RL interface and rules",
            output_dir=output_dir,
            seed=seed,
        )
    )
    tracker = ExperimentTracker(cfg)
    run_dir = tracker.initialize()

    results_file = run_dir / "benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nBenchmark artifacts saved to: {results_file.resolve()}\n")
    return metrics_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL representation latency benchmark.")
    parser.add_argument("--num-positions", "-n", type=int, default=500, help="Number of positions to benchmark")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", "-o", type=str, default="experiments/results", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(num_positions=args.num_positions, seed=args.seed, output_dir=args.output_dir)
