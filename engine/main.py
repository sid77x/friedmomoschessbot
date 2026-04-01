from __future__ import annotations

import argparse
import os

from .evaluation.ensemble import EnsembleEvaluator
from .evaluation.heuristic import HeuristicEvaluator
from .evaluation.ml_model import MLEvaluator
from .evaluation.neural_model import NeuralEvaluator
from .evaluation.positional import PositionalEvaluator
from .evaluation.rl_deep_model import RLDeepEvaluator
from .evaluation.rl_td_model import RLTDLinearEvaluator
from .gui import ChessGUI
from .lichess_bot import LichessBot
from .openings import OpeningBook
from .search import SearchEngine
from .uci import UCIProtocol


def build_engine() -> tuple[SearchEngine, MLEvaluator, NeuralEvaluator, RLTDLinearEvaluator, RLDeepEvaluator]:
    heuristic = HeuristicEvaluator()
    ml_model = MLEvaluator(model_path="models/ml_model.pkl")
    neural_model = NeuralEvaluator(model_path="models/neural_model.pt")
    rl_td_model = RLTDLinearEvaluator(model_path="models/rl_td_model.pkl")
    rl_deep_model = RLDeepEvaluator(model_path="models/rl_deep_model.pt")
    positional = PositionalEvaluator()

    ensemble = EnsembleEvaluator(
        heuristic=heuristic,
        ml_model=ml_model,
        neural_model=neural_model,
        positional_model=positional,
        rl_td_model=rl_td_model,
        rl_deep_model=rl_deep_model,
    )
    openings = OpeningBook.from_builtin()
    search_engine = SearchEngine(evaluator=ensemble, opening_book=openings)
    return search_engine, ml_model, neural_model, rl_td_model, rl_deep_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular Ensemble Chess Engine")
    parser.add_argument("--mode", choices=["gui", "lichess", "train", "uci"], default="gui")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--think-time", type=float, default=1.8, help="Per move think time in seconds")

    parser.add_argument("--token", type=str, default="", help="Lichess API token")

    parser.add_argument("--ml-samples", type=int, default=4500)
    parser.add_argument("--nn-samples", type=int, default=6500)
    parser.add_argument("--nn-epochs", type=int, default=8)
    parser.add_argument("--rl-td-minutes", type=float, default=2.0)
    parser.add_argument("--rl-deep-minutes", type=float, default=2.0)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine, ml_model, neural_model, rl_td_model, rl_deep_model = build_engine()

    if args.mode == "gui":
        app = ChessGUI(engine=engine, depth=args.depth, think_time_s=args.think_time)
        app.run()
        return

    if args.mode == "train":
        print("Training ML model...")
        ml_model.train_synthetic(samples=args.ml_samples)
        print("Saved ML model to models/ml_model.pkl")

        if not args.skip_neural:
            print("Training neural model...")
            neural_model.train_synthetic(
                samples=args.nn_samples,
                epochs=args.nn_epochs,
            )
            print("Saved neural model to models/neural_model.pt")

        if not args.skip_rl:
            print("Training RL TD model...")
            rl_td_model.train_self_play(minutes=args.rl_td_minutes)
            print("Saved RL TD model to models/rl_td_model.pkl")

            print("Training RL deep model...")
            rl_deep_model.train_self_play(minutes=args.rl_deep_minutes)
            print("Saved RL deep model to models/rl_deep_model.pt")
        return

    if args.mode == "lichess":
        token = args.token or os.getenv("LICHESS_BOT_TOKEN", "")
        if not token:
            raise ValueError("Missing token. Pass --token or set LICHESS_BOT_TOKEN")
        bot = LichessBot(token=token, engine=engine)
        bot.run()
        return

    if args.mode == "uci":
        protocol = UCIProtocol(engine)
        protocol.run()


if __name__ == "__main__":
    main()
