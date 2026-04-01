from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import chess

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Always train and save relative to the chess engine project root.
os.chdir(ROOT_DIR)

from engine.evaluation.checkmate_curriculum import CurriculumPosition, build_training_curriculum
from engine.evaluation.heuristic import HeuristicEvaluator
from engine.main import build_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified model trainer (ML + Neural + RL + checkmate drills)")
    parser.add_argument("--minutes", type=float, default=18.0, help="Target training time budget in minutes")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ml-samples", type=int, default=7000)
    parser.add_argument("--nn-samples", type=int, default=9000)
    parser.add_argument("--nn-epochs", type=int, default=10)
    parser.add_argument("--curriculum-repeat", type=int, default=4)
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    return parser.parse_args()


def _clip_label(cp: float) -> float:
    if cp >= 80000:
        return 2500.0
    if cp <= -80000:
        return -2500.0
    return float(max(-2500.0, min(2500.0, cp)))


def _random_positions(samples: int, max_random_plies: int, rng: random.Random) -> Tuple[List[chess.Board], List[float]]:
    heuristic = HeuristicEvaluator()
    boards: List[chess.Board] = []
    labels: List[float] = []

    for _ in range(max(1, samples)):
        board = chess.Board()
        plies = rng.randint(0, max_random_plies)
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        boards.append(board.copy(stack=True))
        labels.append(float(heuristic.evaluate(board)))

    return boards, labels


def _merge_curriculum(
    boards: List[chess.Board],
    labels: List[float],
    curriculum: List[CurriculumPosition],
    repeat: int,
) -> Tuple[List[chess.Board], List[float]]:
    for _ in range(max(1, repeat)):
        for sample in curriculum:
            boards.append(sample.board.copy(stack=True))
            labels.append(_clip_label(sample.target_cp))
    return boards, labels


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    train_start = time.perf_counter()
    deadline = train_start + max(4.0, args.minutes) * 60.0

    engine, ml_model, neural_model, rl_td_model, rl_deep_model = build_engine()
    _ = engine

    print("Building checkmate curriculum (queen/rook/pawn/ladder/knight+bishop + mate in 1/2)...")
    curriculum = build_training_curriculum(seed=args.seed)
    print(f"Curriculum samples: {len(curriculum)}")

    left_min = max(2.0, (deadline - time.perf_counter()) / 60.0)
    scale = max(0.6, min(2.2, left_min / 18.0))
    ml_samples = max(2400, int(args.ml_samples * scale))
    nn_samples = max(3200, int(args.nn_samples * scale))

    boards, labels = _random_positions(samples=ml_samples, max_random_plies=34, rng=rng)
    boards, labels = _merge_curriculum(boards, labels, curriculum, repeat=args.curriculum_repeat)

    if not args.skip_ml:
        print(f"Training ML model on {len(boards)} positions...")
        ml_model.train(boards, labels)
        if Path("models/ml_model.pkl").exists():
            print("Saved models/ml_model.pkl")
        else:
            print("Warning: ML model file was not created.")

    if not args.skip_neural:
        neural_boards, neural_labels = _random_positions(samples=nn_samples, max_random_plies=38, rng=rng)
        neural_boards, neural_labels = _merge_curriculum(
            neural_boards,
            neural_labels,
            curriculum,
            repeat=max(2, args.curriculum_repeat - 1),
        )
        print(f"Training neural model on {len(neural_boards)} positions...")
        neural_model.train(
            positions=neural_boards,
            labels=neural_labels,
            epochs=args.nn_epochs,
            batch_size=128,
            lr=1e-3,
        )
        if Path("models/neural_model.pt").exists():
            print("Saved models/neural_model.pt")
        else:
            print("Warning: Neural model file was not created (Torch may be unavailable).")

    if not args.skip_rl:
        time_left_s = max(90.0, deadline - time.perf_counter())
        td_minutes = max(0.5, (time_left_s * 0.50) / 60.0)
        deep_minutes = max(0.5, (time_left_s * 0.45) / 60.0)

        print(f"Training RL TD model for ~{td_minutes:.1f} minutes...")
        rl_td_model.train_self_play(minutes=td_minutes, curriculum=curriculum, seed=args.seed + 10)
        if Path("models/rl_td_model.pkl").exists():
            print("Saved models/rl_td_model.pkl")
        else:
            print("Warning: RL TD model file was not created.")

        print(f"Training RL deep model for ~{deep_minutes:.1f} minutes...")
        rl_deep_model.train_self_play(minutes=deep_minutes, curriculum=curriculum, seed=args.seed + 20)
        if Path("models/rl_deep_model.pt").exists():
            print("Saved models/rl_deep_model.pt")
        else:
            print("Warning: RL deep model file was not created (Torch may be unavailable).")

    elapsed_min = (time.perf_counter() - train_start) / 60.0
    print(f"Training finished in {elapsed_min:.2f} minutes.")


if __name__ == "__main__":
    main()
