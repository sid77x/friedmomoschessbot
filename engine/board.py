from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import chess


@dataclass(frozen=True)
class MoveRecord:
    uci: str
    san: str
    fen_before: str
    fen_after: str


class EngineBoard:
    """Thin wrapper around python-chess bitboard implementation."""

    def __init__(self, fen: Optional[str] = None) -> None:
        self.board = chess.Board(fen=fen) if fen else chess.Board()
        self.history: List[MoveRecord] = []

    def copy(self) -> "EngineBoard":
        clone = EngineBoard(self.board.fen())
        clone.history = list(self.history)
        return clone

    def fen(self) -> str:
        return self.board.fen()

    def position_key(self) -> str:
        ep = chess.square_name(self.board.ep_square) if self.board.ep_square is not None else "-"
        return f"{self.board.board_fen()} {self.board.turn} {self.board.castling_xfen()} {ep}"

    def legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        fen_before = self.board.fen()
        san = self.board.san(move)
        self.board.push(move)
        self.history.append(MoveRecord(move.uci(), san, fen_before, self.board.fen()))

    def push_uci(self, move_uci: str) -> chess.Move:
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")
        self.push(move)
        return move

    def pop(self) -> chess.Move:
        move = self.board.pop()
        if self.history:
            self.history.pop()
        return move

    def is_game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def result(self) -> str:
        return self.board.result(claim_draw=True)

    @property
    def turn(self) -> chess.Color:
        return self.board.turn
