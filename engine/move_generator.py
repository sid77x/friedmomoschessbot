from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import chess


PIECE_ORDER = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


class MoveGenerator:
    def legal_moves(self, board: chess.Board) -> List[chess.Move]:
        return list(board.legal_moves)

    def ordered_moves(
        self,
        board: chess.Board,
        moves: Optional[Sequence[chess.Move]] = None,
        tt_move: Optional[chess.Move] = None,
        killers: Optional[Iterable[chess.Move]] = None,
        history_scores: Optional[Dict[str, int]] = None,
    ) -> List[chess.Move]:
        if moves is None:
            moves = list(board.legal_moves)

        killer_set = {m.uci() for m in killers if m is not None} if killers else set()
        history_scores = history_scores or {}

        def score(move: chess.Move) -> int:
            if tt_move and move == tt_move:
                return 1_000_000

            total = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                victim_score = PIECE_ORDER.get(victim.piece_type, 0) if victim else 0
                attacker_score = PIECE_ORDER.get(attacker.piece_type, 0) if attacker else 0
                total += 20_000 + (victim_score * 12 - attacker_score)
            if move.promotion:
                total += 12_000 + PIECE_ORDER.get(move.promotion, 0)
            if move.uci() in killer_set:
                total += 5_000
            total += history_scores.get(move.uci(), 0)

            board.push(move)
            if board.is_checkmate():
                total += 900_000
            elif board.is_stalemate():
                total -= 600_000
            if board.is_check():
                total += 700
            board.pop()
            return total

        return sorted(moves, key=score, reverse=True)
