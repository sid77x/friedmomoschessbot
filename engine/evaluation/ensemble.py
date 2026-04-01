from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import chess

from .features import game_phase
from .heuristic import HeuristicEvaluator
from .ml_model import MLEvaluator
from .neural_model import NeuralEvaluator
from .positional import PositionalEvaluator
from .rl_deep_model import RLDeepEvaluator
from .rl_td_model import RLTDLinearEvaluator


@dataclass
class EnsembleBreakdown:
    phase: str
    raw: Dict[str, float]
    weights: Dict[str, float]
    final_score: float


class EnsembleEvaluator:
    def __init__(
        self,
        heuristic: HeuristicEvaluator | None = None,
        ml_model: MLEvaluator | None = None,
        neural_model: NeuralEvaluator | None = None,
        positional_model: PositionalEvaluator | None = None,
        rl_td_model: RLTDLinearEvaluator | None = None,
        rl_deep_model: RLDeepEvaluator | None = None,
    ) -> None:
        self.heuristic = heuristic or HeuristicEvaluator()
        self.ml_model = ml_model or MLEvaluator()
        self.neural_model = neural_model or NeuralEvaluator()
        self.positional_model = positional_model or PositionalEvaluator()
        self.rl_td_model = rl_td_model or RLTDLinearEvaluator()
        self.rl_deep_model = rl_deep_model or RLDeepEvaluator()

        self.phase_weights: Dict[str, Dict[str, float]] = {
            "opening": {
                "heuristic": 0.40,
                "ml": 0.18,
                "neural": 0.08,
                "positional": 0.16,
                "rl_td": 0.12,
                "rl_deep": 0.06,
            },
            "midgame": {
                "heuristic": 0.24,
                "ml": 0.18,
                "neural": 0.20,
                "positional": 0.12,
                "rl_td": 0.14,
                "rl_deep": 0.12,
            },
            "endgame": {
                "heuristic": 0.14,
                "ml": 0.18,
                "neural": 0.24,
                "positional": 0.10,
                "rl_td": 0.16,
                "rl_deep": 0.18,
            },
        }

    def evaluate(self, board: chess.Board, return_breakdown: bool = False) -> float | Tuple[float, EnsembleBreakdown]:
        phase = game_phase(board)
        base_w = self.phase_weights[phase]

        raw_scores = {
            "heuristic": self.heuristic.evaluate(board),
            "ml": self.ml_model.evaluate(board),
            "neural": self.neural_model.evaluate(board),
            "positional": self.positional_model.evaluate(board),
            "rl_td": self.rl_td_model.evaluate(board),
            "rl_deep": self.rl_deep_model.evaluate(board),
        }

        availability = {
            "heuristic": True,
            "ml": self.ml_model.is_ready,
            "neural": self.neural_model.is_ready,
            "positional": True,
            "rl_td": self.rl_td_model.is_ready,
            "rl_deep": self.rl_deep_model.is_ready,
        }

        active_weight_sum = sum(base_w[name] for name, ready in availability.items() if ready)
        if active_weight_sum <= 0.0:
            active_weight_sum = 1.0

        w = {
            name: (base_w[name] / active_weight_sum if availability[name] else 0.0)
            for name in base_w
        }

        final_score = (
            w["heuristic"] * raw_scores["heuristic"]
            + w["ml"] * raw_scores["ml"]
            + w["neural"] * raw_scores["neural"]
            + w["positional"] * raw_scores["positional"]
            + w["rl_td"] * raw_scores["rl_td"]
            + w["rl_deep"] * raw_scores["rl_deep"]
        )

        if return_breakdown:
            return final_score, EnsembleBreakdown(phase=phase, raw=raw_scores, weights=w, final_score=final_score)
        return final_score
