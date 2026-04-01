from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import chess
import numpy as np

from .features import extract_feature_dict, extract_feature_vector
from .heuristic import HeuristicEvaluator

try:
    from sklearn.ensemble import RandomForestRegressor

    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestRegressor = None
    SKLEARN_AVAILABLE = False


@dataclass
class MLModelMeta:
    backend: str
    feature_names: List[str]


class MLEvaluator:
    def __init__(self, model_path: str = "models/ml_model.pkl") -> None:
        self.model_path = model_path
        self.meta: MLModelMeta | None = None
        self.model = None
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self._load_if_available()

    @property
    def is_ready(self) -> bool:
        return self.model is not None or self.weights is not None

    def _load_if_available(self) -> None:
        if not os.path.exists(self.model_path):
            return
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)

        self.meta = MLModelMeta(
            backend=payload.get("backend", "linear"),
            feature_names=payload.get("feature_names", []),
        )
        if self.meta.backend == "sklearn":
            self.model = payload.get("model")
        else:
            self.weights = payload.get("weights")
            self.bias = float(payload.get("bias", 0.0))

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        payload = {
            "backend": self.meta.backend if self.meta else "linear",
            "feature_names": self.meta.feature_names if self.meta else [],
            "model": self.model,
            "weights": self.weights,
            "bias": self.bias,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)

    def _features_compatible(self, new_names: List[str]) -> bool:
        return bool(self.meta and self.meta.feature_names and self.meta.feature_names == new_names)

    def train(self, positions: Sequence[chess.Board], labels: Sequence[float]) -> None:
        vectors = [extract_feature_vector(b) for b in positions]
        X = np.array([v.values for v in vectors], dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        if X.size == 0:
            return

        new_feature_names = vectors[0].names if vectors else []

        # Incremental path: if an existing sklearn RF model uses the same feature
        # schema, warm-start and grow the forest instead of resetting it.
        can_increment_rf = (
            SKLEARN_AVAILABLE
            and self.meta is not None
            and self.meta.backend == "sklearn"
            and self.model is not None
            and isinstance(self.model, RandomForestRegressor)
            and self._features_compatible(new_feature_names)
        )

        if can_increment_rf:
            current_trees = int(getattr(self.model, "n_estimators", 120))
            self.model.set_params(warm_start=True, n_estimators=current_trees + 48)
            self.model.fit(X, y)
            self.meta = MLModelMeta(backend="sklearn", feature_names=new_feature_names)
            self._save()
            return

        self.meta = MLModelMeta(
            backend="sklearn" if SKLEARN_AVAILABLE else "linear",
            feature_names=new_feature_names,
        )

        if SKLEARN_AVAILABLE:
            self.model = RandomForestRegressor(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=3,
                random_state=7,
                n_jobs=-1,
                warm_start=False,
            )
            self.model.fit(X, y)
        else:
            can_increment_linear = (
                self.weights is not None
                and self._features_compatible(new_feature_names)
                and len(self.weights) == X.shape[1]
            )

            if can_increment_linear:
                lr = 8e-4
                for _ in range(6):
                    pred = X @ self.weights + self.bias
                    err = y - pred
                    self.weights += lr * (X.T @ err) / max(1, len(X))
                    self.bias += float(lr * np.mean(err))
            else:
                X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
                w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
                self.weights = w[:-1]
                self.bias = float(w[-1])

        self._save()

    def _vector_for_meta(self, board: chess.Board) -> np.ndarray:
        feature_map = extract_feature_dict(board)
        feature_map["side_to_move"] = 1.0 if board.turn == chess.WHITE else -1.0

        if self.meta and self.meta.feature_names:
            names = self.meta.feature_names
        else:
            names = extract_feature_vector(board).names
        return np.array([feature_map.get(name, 0.0) for name in names], dtype=np.float32)

    def train_synthetic(self, samples: int = 4000, max_random_plies: int = 30) -> None:
        heuristic = HeuristicEvaluator()
        boards: List[chess.Board] = []
        labels: List[float] = []

        for _ in range(samples):
            b = chess.Board()
            plies = random.randint(0, max_random_plies)
            for _ in range(plies):
                if b.is_game_over(claim_draw=True):
                    break
                b.push(random.choice(list(b.legal_moves)))
            boards.append(b.copy())
            labels.append(float(heuristic.evaluate(b)))

        self.train(boards, labels)

    def evaluate(self, board: chess.Board) -> float:
        vec = self._vector_for_meta(board).reshape(1, -1)
        if self.meta and self.meta.backend == "sklearn" and self.model is not None:
            return float(self.model.predict(vec)[0])
        if self.weights is not None:
            return float(np.dot(vec[0], self.weights) + self.bias)
        return 0.0
