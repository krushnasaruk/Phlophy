"""
100-Game Random Self-Play and Environment Determinism Test Suite.
Validates environment stability, zero illegal actions, and strict reproducibility.
"""

import pytest
from chess_rl.chess_env.environments.standard_chess import StandardChessEnvironment
from chess_rl.chess_env.base import Player
from chess_rl.chess_env.rules.move_generator import MoveGenerator
from chess_rl.agents.policies.random_masked_agent import RandomMaskedAgent


def test_100_random_self_play_games():
    """
    Simulate 100 complete games of Random-vs-Random masked play.
    Strictly asserts 0 illegal actions and 0 runtime exceptions.
    """
    num_games = 100
    white_wins = 0
    black_wins = 0
    draws = 0
    total_plies = 0
    max_plies = 0
    illegal_actions = 0

    for game_idx in range(num_games):
        env = StandardChessEnvironment()
        agent_white = RandomMaskedAgent(seed=1000 + game_idx * 2)
        agent_black = RandomMaskedAgent(seed=1000 + game_idx * 2 + 1)

        res = env.reset()
        ply_count = 0

        while not env.is_terminal() and ply_count < 200:
            current_agent = agent_white if env.current_player == Player.WHITE else agent_black
            legal_before = env.legal_actions()

            try:
                decision = current_agent.select_move(env)
            except Exception as e:
                pytest.fail(f"Agent raised exception during move selection: {e}")

            # Verify chosen move is legal
            if decision.selected_move not in legal_before:
                illegal_actions += 1
                pytest.fail(f"Illegal move chosen: {decision.selected_move} in position: {env.fen()}")

            env.step(decision.selected_move)
            ply_count += 1

        total_plies += ply_count
        max_plies = max(max_plies, ply_count)

        # Terminal evaluation
        reward_white = env.get_reward(Player.WHITE)
        if reward_white > 0:
            white_wins += 1
        elif reward_white < 0:
            black_wins += 1
        else:
            draws += 1

    avg_plies = total_plies / num_games
    print(
        f"\n[100 Random Self-Play Games Summary]\n"
        f"  Total Games: {num_games}\n"
        f"  White Wins:  {white_wins}\n"
        f"  Black Wins:  {black_wins}\n"
        f"  Draws:       {draws}\n"
        f"  Avg Plies:   {avg_plies:.1f}\n"
        f"  Max Plies:   {max_plies}\n"
        f"  Illegal Moves: {illegal_actions}"
    )

    assert illegal_actions == 0
    assert (white_wins + black_wins + draws) == num_games


def test_environment_determinism():
    """
    Verify that identical seeds and move sequences produce bitwise-identical game states.
    """
    seed = 42

    def run_sim(s):
        env = StandardChessEnvironment()
        agent_w = RandomMaskedAgent(seed=s)
        agent_b = RandomMaskedAgent(seed=s + 1)
        res = env.reset()
        fens = [env.fen()]
        moves = []
        for _ in range(30):
            if env.is_terminal():
                break
            cur = agent_w if env.current_player == Player.WHITE else agent_b
            dec = cur.select_move(env)
            moves.append(dec.selected_move.to_uci())
            env.step(dec.selected_move)
            fens.append(env.fen())
        return fens, moves

    fens_run1, moves_run1 = run_sim(seed)
    fens_run2, moves_run2 = run_sim(seed)

    assert moves_run1 == moves_run2
    assert fens_run1 == fens_run2
