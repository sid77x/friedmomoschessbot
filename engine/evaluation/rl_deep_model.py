from __future__ import annotations

import os
import random
import time
import importlib
from typing import List, Sequence, Tuple

import chess
import numpy as np

from .checkmate_curriculum import CurriculumPosition
from .features import extract_feature_dict, extract_feature_vector

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DeepRLValueNet(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 192),
                nn.ReLU(),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Linear(96, 1),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x)


class RLDeepEvaluator:
    def __init__(self, model_path: str = "models/rl_deep_model.pt") -> None:
        self.model_path = model_path
        self.feature_names = extract_feature_vector(chess.Board()).names
        self.input_dim = len(self.feature_names)
        self.model = None
        self._has_trained_weights = False

        if TORCH_AVAILABLE:
            self.model = DeepRLValueNet(self.input_dim)
            self._load_if_available()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self._has_trained_weights

    def _load_if_available(self) -> None:
        if self.model is None or not os.path.exists(self.model_path):
            return
        state = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self._has_trained_weights = True

    def _save(self) -> None:
        if self.model is None:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        self._has_trained_weights = True

    def _vector(self, board: chess.Board) -> np.ndarray:
        fmap = extract_feature_dict(board)
        fmap["side_to_move"] = 1.0 if board.turn == chess.WHITE else -1.0
        return np.array([fmap.get(name, 0.0) for name in self.feature_names], dtype=np.float32)

    def _predict_raw(self, board: chess.Board) -> float:
        if self.model is None:
            return 0.0
        x = torch.tensor(self._vector(board), dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            return float(self.model(x).item())

    def evaluate(self, board: chess.Board) -> float:
        if not self._has_trained_weights:
            return 0.0
        return float(900.0 * self._predict_raw(board))

    def _result_white_value(self, board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    def _choose_move(self, board: chess.Board, epsilon: float, rng: random.Random) -> chess.Move:
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]
        if rng.random() < epsilon or self.model is None:
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

    def _fit(self, examples: Sequence[Tuple[np.ndarray, float]], epochs: int = 2, lr: float = 8e-4) -> None:
        if self.model is None or not examples:
            return

        X = np.stack([e[0] for e in examples]).astype(np.float32)
        y = np.array([e[1] for e in examples], dtype=np.float32).reshape(-1, 1)

        tx = torch.tensor(X, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.float32)

        ds = torch.utils.data.TensorDataset(tx, ty)
        loader = torch.utils.data.DataLoader(ds, batch_size=min(128, len(ds)), shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()

        for _ in range(max(1, epochs)):
            for bx, by in loader:
                pred = self.model(bx)
                loss = loss_fn(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()

    def train_self_play(
        self,
        minutes: float = 4.0,
        curriculum: Sequence[CurriculumPosition] | None = None,
        gamma: float = 0.98,
        epsilon: float = 0.20,
        seed: int = 11,
    ) -> None:
        if not TORCH_AVAILABLE or self.model is None:
            return

        rng = random.Random(seed)
        deadline = time.perf_counter() + max(10.0, minutes * 60.0)

        examples: List[Tuple[np.ndarray, float]] = []
        curr = list(curriculum or [])
        local_epsilon = epsilon

        while time.perf_counter() < deadline:
            board = chess.Board()
            for _ in range(rng.randint(0, 10)):
                if board.is_game_over(claim_draw=True):
                    break
                board.push(rng.choice(list(board.legal_moves)))

            history: List[np.ndarray] = []
            max_plies = 110
            for _ in range(max_plies):
                if board.is_game_over(claim_draw=True):
                    break
                history.append(self._vector(board))
                move = self._choose_move(board, local_epsilon, rng)
                board.push(move)

            result = self._result_white_value(board)
            if not board.is_game_over(claim_draw=True):
                result = float(np.clip(self._predict_raw(board), -1.0, 1.0))

            for idx, vec in enumerate(history):
                discount = gamma ** max(0, len(history) - idx - 1)
                target = float(np.clip(result * discount, -1.0, 1.0))
                examples.append((vec, target))

            if curr and rng.random() < 0.60:
                sample = rng.choice(curr)
                examples.append((self._vector(sample.board), float(np.clip(sample.target_cp / 1000.0, -1.0, 1.0))))

            if len(examples) >= 800:
                self._fit(examples, epochs=2)
                examples.clear()

            local_epsilon = max(0.06, local_epsilon * 0.999)

        if examples:
            self._fit(examples, epochs=2)

        self._save()
