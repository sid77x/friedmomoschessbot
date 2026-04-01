from __future__ import annotations

import chess

from .features import extract_feature_dict


class PositionalEvaluator:
    """Activity, center control, and space-focused heuristic."""

    def evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -100000.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        f = extract_feature_dict(board)
        score = (
            14.0 * f["mobility"]
            + 16.0 * f["center_control"]
            + 20.0 * f["space"]
            + 2.0 * f["king_safety"]
            + 10.0 * f["development"]
            + 12.0 * f["king_activity"]
            + 10.0 * f["king_confinement"]
            - 10.0 * f["early_queen_penalty"]
            - 0.20 * f["hanging_material"]
            + 0.08 * f["tactical_pressure"]
            + 10.0 * f["pawn_passed"]
            + 2.0 * f["pawn_passed_advance"]
        )
        return score
