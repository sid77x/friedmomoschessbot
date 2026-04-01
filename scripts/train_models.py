from __future__ import annotations

import argparse
from collections import Counter
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import chess
import chess.pgn

ROOT_DIR = Path(__file__).resolve().parents[1]
LAUNCH_CWD = Path.cwd()
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
    parser.add_argument("--pgn", type=str, default="", help="Path to a PGN file with strong/master games")
    parser.add_argument("--min-elo", type=int, default=2200, help="Minimum WhiteElo and BlackElo to include a PGN game")
    parser.add_argument("--max-pgn-games", type=int, default=1500, help="Maximum PGN games to read (0 = no limit)")
    parser.add_argument("--pgn-sample-every", type=int, default=4, help="Take one position every N plies from each PGN game")
    parser.add_argument("--max-pgn-positions", type=int, default=12000, help="Maximum positions sampled from PGN")
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


def _result_to_white_cp(result: str) -> float | None:
    if result == "1-0":
        return 900.0
    if result == "0-1":
        return -900.0
    if result == "1/2-1/2":
        return 0.0
    return None


def _safe_header_int(headers: chess.pgn.Headers, key: str) -> int:
    raw = headers.get(key, "0")
    try:
        return int(raw)
    except Exception:
        return 0


def _positions_from_master_pgn(
    pgn_path: Path,
    min_elo: int,
    max_games: int,
    sample_every: int,
    max_positions: int,
) -> Tuple[List[chess.Board], List[float]]:
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    heuristic = HeuristicEvaluator()
    boards: List[chess.Board] = []
    labels: List[float] = []

    games_read = 0
    games_used = 0

    with pgn_path.open("r", encoding="utf-8", errors="ignore") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            games_read += 1
            if max_games > 0 and games_read > max_games:
                break

            result = _result_to_white_cp(game.headers.get("Result", "*"))
            if result is None:
                continue

            white_elo = _safe_header_int(game.headers, "WhiteElo")
            black_elo = _safe_header_int(game.headers, "BlackElo")
            if white_elo < min_elo or black_elo < min_elo:
                continue

            games_used += 1
            board = game.board()

            for ply_idx, move in enumerate(game.mainline_moves(), start=1):
                if ply_idx % max(1, sample_every) == 0:
                    # Convert white-centric result to side-to-move perspective.
                    stm_result = result if board.turn == chess.WHITE else -result
                    h = float(max(-900.0, min(900.0, heuristic.evaluate(board))))
                    blended = 0.72 * stm_result + 0.28 * h

                    boards.append(board.copy(stack=True))
                    labels.append(float(max(-1200.0, min(1200.0, blended))))

                    if len(boards) >= max_positions:
                        break

                board.push(move)

            if len(boards) >= max_positions:
                break

    print(
        f"Loaded {len(boards)} positions from {games_used} master games "
        f"(read {games_read} games, min_elo={min_elo})."
    )
    return boards, labels


def _resolve_pgn_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        Path(raw_path),
        ROOT_DIR / raw_path,
        ROOT_DIR.parent / raw_path,
        LAUNCH_CWD / raw_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return p


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    train_start = time.perf_counter()
    deadline = train_start + max(4.0, args.minutes) * 60.0

    engine, ml_model, neural_model, rl_td_model, rl_deep_model = build_engine()
    _ = engine

    print("Building mixed curriculum (checkmates + tactics + opening/endgame principles)...")
    curriculum = build_training_curriculum(seed=args.seed)
    print(f"Curriculum samples: {len(curriculum)}")
    prefix_counts = Counter(sample.tag.split("_")[0] for sample in curriculum)
    print(f"Curriculum breakdown: {dict(prefix_counts)}")

    left_min = max(2.0, (deadline - time.perf_counter()) / 60.0)
    scale = max(0.6, min(2.2, left_min / 18.0))
    ml_samples = max(2400, int(args.ml_samples * scale))
    nn_samples = max(3200, int(args.nn_samples * scale))

    boards, labels = _random_positions(samples=ml_samples, max_random_plies=34, rng=rng)
    boards, labels = _merge_curriculum(boards, labels, curriculum, repeat=args.curriculum_repeat)

    if args.pgn:
        pgn_path = _resolve_pgn_path(args.pgn)
        pgn_boards, pgn_labels = _positions_from_master_pgn(
            pgn_path=pgn_path,
            min_elo=args.min_elo,
            max_games=args.max_pgn_games,
            sample_every=args.pgn_sample_every,
            max_positions=args.max_pgn_positions,
        )
        boards.extend(pgn_boards)
        labels.extend(pgn_labels)

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

        if args.pgn:
            # Reuse PGN supervision for neural model as well.
            pgn_path = _resolve_pgn_path(args.pgn)
            pgn_boards, pgn_labels = _positions_from_master_pgn(
                pgn_path=pgn_path,
                min_elo=args.min_elo,
                max_games=max(1, args.max_pgn_games // 2),
                sample_every=max(1, args.pgn_sample_every),
                max_positions=max(1000, args.max_pgn_positions // 2),
            )
            neural_boards.extend(pgn_boards)
            neural_labels.extend(pgn_labels)

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
