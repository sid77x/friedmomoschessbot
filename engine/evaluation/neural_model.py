from __future__ import annotations

import os
import random
from typing import List, Sequence

import chess
import numpy as np

from .heuristic import HeuristicEvaluator

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def encode_board(board: chess.Board) -> np.ndarray:
    arr = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        idx = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        rank = chess.square_rank(sq)
        file_ = chess.square_file(sq)
        arr[idx, rank, file_] = 1.0

    extras = np.array(
        [
            1.0 if board.turn == chess.WHITE else -1.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            min(board.halfmove_clock, 100) / 100.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([arr.flatten(), extras])


if TORCH_AVAILABLE:

    class PositionNet(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class NeuralEvaluator:
    def __init__(self, model_path: str = "models/neural_model.pt") -> None:
        self.model_path = model_path
        self.model = None
        self._has_trained_weights = False
        self.input_dim = len(encode_board(chess.Board()))
        if TORCH_AVAILABLE:
            self.model = PositionNet(self.input_dim)
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

    def train(
        self,
        positions: Sequence[chess.Board],
        labels: Sequence[float],
        epochs: int = 8,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> None:
        if not TORCH_AVAILABLE or self.model is None:
            return
        if len(positions) == 0 or len(labels) == 0:
            return

        X = [encode_board(b) for b in positions]
        y = np.array(labels, dtype=np.float32)

        tensor_x = torch.tensor(np.stack(X), dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()

        self.model.train()
        for _ in range(max(1, epochs)):
            for bx, by in loader:
                pred = self.model(bx)
                loss = loss_fn(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()
        self._save()

    def train_synthetic(
        self,
        samples: int = 6000,
        max_random_plies: int = 36,
        epochs: int = 8,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> None:
        if not TORCH_AVAILABLE or self.model is None:
            return

        heuristic = HeuristicEvaluator()
        boards: List[chess.Board] = []
        y: List[float] = []

        for _ in range(samples):
            b = chess.Board()
            plies = random.randint(0, max_random_plies)
            for _ in range(plies):
                if b.is_game_over(claim_draw=True):
                    break
                b.push(random.choice(list(b.legal_moves)))
            boards.append(b.copy(stack=True))
            y.append(float(heuristic.evaluate(b)))

        self.train(
            positions=boards,
            labels=y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

    def evaluate(self, board: chess.Board) -> float:
        if self.model is None or not self._has_trained_weights:
            return 0.0
        x = torch.tensor(encode_board(board), dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            return float(self.model(x).item())
