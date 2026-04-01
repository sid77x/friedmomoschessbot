from __future__ import annotations

import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import List, Sequence

import chess
import numpy as np

from .checkmate_curriculum import CurriculumPosition
from .features import extract_feature_dict, extract_feature_vector


@dataclass
class RLTDMeta:
    feature_names: List[str]


class RLTDLinearEvaluator:
    def __init__(self, model_path: str = "models/rl_td_model.pkl") -> None:
        self.model_path = model_path
        self.meta = RLTDMeta(feature_names=extract_feature_vector(chess.Board()).names)
        self.weights = np.zeros(len(self.meta.feature_names), dtype=np.float32)
        self.bias = 0.0
        self._has_trained_weights = False
        self._load_if_available()

    @property
    def is_ready(self) -> bool:
        return self.weights is not None and self._has_trained_weights

    def _load_if_available(self) -> None:
        if not os.path.exists(self.model_path):
            return
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)

        names = payload.get("feature_names", self.meta.feature_names)
        weights = payload.get("weights")
        bias = float(payload.get("bias", 0.0))

        if weights is None:
            return

        self.meta = RLTDMeta(feature_names=list(names))
        self.weights = np.array(weights, dtype=np.float32)
        self.bias = bias
        self._has_trained_weights = True

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        payload = {
            "feature_names": self.meta.feature_names,
            "weights": self.weights,
            "bias": self.bias,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)
        self._has_trained_weights = True

    def _vector(self, board: chess.Board) -> np.ndarray:
        fmap = extract_feature_dict(board)
        fmap["side_to_move"] = 1.0 if board.turn == chess.WHITE else -1.0
        scales = {
            "material": 2500.0,
            "mobility": 40.0,
            "center_control": 20.0,
            "space": 20.0,
            "king_safety": 200.0,
            "development": 8.0,
            "early_queen_penalty": 2.0,
            "king_activity": 8.0,
            "king_confinement": 8.0,
            "hanging_material": 1000.0,
            "tactical_pressure": 100.0,
            "pin_pressure": 30.0,
            "skewer_pressure": 30.0,
            "fork_pressure": 30.0,
            "pawn_doubled": 8.0,
            "pawn_isolated": 8.0,
            "pawn_passed": 8.0,
            "pawn_passed_advance": 48.0,
            "side_to_move": 1.0,
        }

        values = []
        for name in self.meta.feature_names:
            raw = float(fmap.get(name, 0.0))
            scale = scales.get(name, 100.0)
            values.append(float(np.clip(raw / scale, -4.0, 4.0)))
        return np.array(values, dtype=np.float32)

    def _predict_raw_from_vec(self, vec: np.ndarray) -> float:
        value = float(np.dot(vec, self.weights) + self.bias)
        return float(np.clip(value, -8.0, 8.0))

    def _predict_raw(self, board: chess.Board) -> float:
        return self._predict_raw_from_vec(self._vector(board))

    def _update_towards(self, board: chess.Board, target: float, alpha: float) -> None:
        vec = self._vector(board)
        pred = self._predict_raw_from_vec(vec)
        err = float(np.clip(target - pred, -2.0, 2.0))
        self.weights += alpha * err * vec
        self.bias += alpha * err
        self.weights = np.clip(self.weights, -10.0, 10.0)
        self.bias = float(np.clip(self.bias, -5.0, 5.0))

    def evaluate(self, board: chess.Board) -> float:
        value = np.tanh(self._predict_raw(board))
        return float(900.0 * value)

    def _material(self, board: chess.Board) -> float:
        return float(extract_feature_dict(board)["material"])

    def _result_white_value(self, board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    def _choose_move(self, board: chess.Board, epsilon: float, rng: random.Random) -> chess.Move:
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]
        if rng.random() < epsilon:
            return rng.choice(legal)

        best_move = legal[0]
        if board.turn == chess.WHITE:
            best_score = -1e9
            for move in legal:
                board.push(move)
                score = self._predict_raw(board)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
        else:
            best_score = 1e9
            for move in legal:
                board.push(move)
                score = self._predict_raw(board)
                board.pop()
                if score < best_score:
                    best_score = score
                    best_move = move
        return best_move

    def train_self_play(
        self,
        minutes: float = 4.0,
        curriculum: Sequence[CurriculumPosition] | None = None,
        alpha: float = 0.012,
        gamma: float = 0.97,
        epsilon: float = 0.18,
        max_game_plies: int = 100,
        seed: int = 7,
    ) -> None:
        rng = random.Random(seed)
        deadline = time.perf_counter() + max(10.0, minutes * 60.0)

        curr = list(curriculum or [])
        local_epsilon = epsilon

        while time.perf_counter() < deadline:
            if curr and rng.random() < 0.25:
                sample = rng.choice(curr)
                target = float(np.clip(sample.target_cp / 1000.0, -1.0, 1.0))
                self._update_towards(sample.board, target, alpha * 1.3)
                continue

            board = chess.Board()
            for _ in range(rng.randint(0, 8)):
                if board.is_game_over(claim_draw=True):
                    break
                board.push(rng.choice(list(board.legal_moves)))

            for _ in range(max_game_plies):
                if board.is_game_over(claim_draw=True):
                    break

                vec = self._vector(board)
                before_material = self._material(board)
                move = self._choose_move(board, local_epsilon, rng)
                board.push(move)

                after_material = self._material(board)
                shaped_reward = 0.0015 * (after_material - before_material)
                terminal_reward = self._result_white_value(board) if board.is_game_over(claim_draw=True) else 0.0
                reward = shaped_reward + terminal_reward

                pred = self._predict_raw_from_vec(vec)
                next_value = 0.0 if board.is_game_over(claim_draw=True) else self._predict_raw(board)
                target = float(np.clip(reward + gamma * next_value, -1.0, 1.0))

                err = float(np.clip(target - pred, -2.0, 2.0))
                self.weights += alpha * err * vec
                self.bias += alpha * err
                self.weights = np.clip(self.weights, -10.0, 10.0)
                self.bias = float(np.clip(self.bias, -5.0, 5.0))

                if board.is_game_over(claim_draw=True):
                    break

            local_epsilon = max(0.05, local_epsilon * 0.9995)

        self._save()
