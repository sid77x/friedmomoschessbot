from __future__ import annotations

import math
from typing import Dict

import chess

from .features import extract_feature_dict, game_phase


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

KING_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]

PST = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


class HeuristicEvaluator:
    def evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -100000.0 if board.turn == chess.WHITE else 100000.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0

        score = self._material_and_pst(board)
        score += self._structure_and_safety(board)
        score += self._phase_objectives(board)
        score += self._special_endgame_mating_patterns(board)

        return score if board.turn == chess.WHITE else -score

    def _material_and_pst(self, board: chess.Board) -> float:
        score = 0.0
        for piece_type, value in PIECE_VALUES.items():
            for sq in board.pieces(piece_type, chess.WHITE):
                score += value + PST[piece_type][sq]
            for sq in board.pieces(piece_type, chess.BLACK):
                mirrored = chess.square_mirror(sq)
                score -= value + PST[piece_type][mirrored]
        return score

    def _structure_and_safety(self, board: chess.Board) -> float:
        f: Dict[str, float] = extract_feature_dict(board)
        return (
            8.0 * f["mobility"]
            + 12.0 * f["center_control"]
            + 8.0 * f["space"]
            + 1.5 * f["king_safety"]
            + 0.08 * f["tactical_pressure"]
            - 0.30 * f["hanging_material"]
            - 18.0 * f["pawn_doubled"]
            - 14.0 * f["pawn_isolated"]
            + 18.0 * f["pawn_passed"]
            + 3.0 * f["pawn_passed_advance"]
        )

    def _phase_objectives(self, board: chess.Board) -> float:
        phase = game_phase(board)
        f: Dict[str, float] = extract_feature_dict(board)

        if phase == "opening":
            return (
                18.0 * f["development"]
                + 10.0 * f["center_control"]
                + 8.0 * f["king_safety"]
                - 24.0 * f["early_queen_penalty"]
            )

        if phase == "midgame":
            return (
                16.0 * f["mobility"]
                + 10.0 * f["center_control"]
                + 10.0 * f["king_confinement"]
                + 0.12 * f["tactical_pressure"]
            )

        return (
            24.0 * f["king_activity"]
            + 14.0 * f["king_confinement"]
            + 26.0 * f["pawn_passed"]
            + 4.5 * f["pawn_passed_advance"]
        )

    def _special_endgame_mating_patterns(self, board: chess.Board) -> float:
        phase = game_phase(board)
        if phase != "endgame":
            return 0.0

        score = 0.0
        score += self._kqk_objective(board)
        score += self._krk_objective(board)
        score += self._kbnk_objective(board)
        score += self._kpk_objective(board)
        score += self._ladder_mate_objective(board)
        return score

    def _has_only(self, board: chess.Board, color: chess.Color, pieces: Dict[int, int]) -> bool:
        counts: Dict[int, int] = {}
        for _, piece in board.piece_map().items():
            if piece.color != color:
                continue
            if piece.piece_type == chess.KING:
                continue
            counts[piece.piece_type] = counts.get(piece.piece_type, 0) + 1
        return counts == pieces

    def _is_lone_king(self, board: chess.Board, color: chess.Color) -> bool:
        return self._has_only(board, color, {})

    def _edge_bonus(self, sq: chess.Square) -> float:
        rank = chess.square_rank(sq)
        file_ = chess.square_file(sq)
        center_dist = abs(rank - 3.5) + abs(file_ - 3.5)
        return center_dist * 10.0

    def _king_distance(self, a: chess.Square | None, b: chess.Square | None) -> float:
        if a is None or b is None:
            return 0.0
        ar, af = chess.square_rank(a), chess.square_file(a)
        br, bf = chess.square_rank(b), chess.square_file(b)
        return float(abs(ar - br) + abs(af - bf))

    def _drive_king_score(self, winner: chess.Color, loser: chess.Color, board: chess.Board, base: float) -> float:
        loser_king = board.king(loser)
        winner_king = board.king(winner)
        if loser_king is None or winner_king is None:
            return 0.0
        edge = self._edge_bonus(loser_king)
        king_support = 14.0 - self._king_distance(winner_king, loser_king)
        signed = base + edge + 6.0 * king_support
        return signed if winner == chess.WHITE else -signed

    def _kqk_objective(self, board: chess.Board) -> float:
        if self._is_lone_king(board, chess.BLACK) and self._has_only(board, chess.WHITE, {chess.QUEEN: 1}):
            return self._drive_king_score(chess.WHITE, chess.BLACK, board, base=160.0)
        if self._is_lone_king(board, chess.WHITE) and self._has_only(board, chess.BLACK, {chess.QUEEN: 1}):
            return self._drive_king_score(chess.BLACK, chess.WHITE, board, base=160.0)
        return 0.0

    def _krk_objective(self, board: chess.Board) -> float:
        if self._is_lone_king(board, chess.BLACK) and self._has_only(board, chess.WHITE, {chess.ROOK: 1}):
            return self._drive_king_score(chess.WHITE, chess.BLACK, board, base=120.0)
        if self._is_lone_king(board, chess.WHITE) and self._has_only(board, chess.BLACK, {chess.ROOK: 1}):
            return self._drive_king_score(chess.BLACK, chess.WHITE, board, base=120.0)
        return 0.0

    def _kbnk_objective(self, board: chess.Board) -> float:
        if self._is_lone_king(board, chess.BLACK) and self._has_only(board, chess.WHITE, {chess.BISHOP: 1, chess.KNIGHT: 1}):
            return self._kbnk_drive_corner(chess.WHITE, chess.BLACK, board)
        if self._is_lone_king(board, chess.WHITE) and self._has_only(board, chess.BLACK, {chess.BISHOP: 1, chess.KNIGHT: 1}):
            return self._kbnk_drive_corner(chess.BLACK, chess.WHITE, board)
        return 0.0

    def _kbnk_drive_corner(self, winner: chess.Color, loser: chess.Color, board: chess.Board) -> float:
        bishops = board.pieces(chess.BISHOP, winner)
        loser_king = board.king(loser)
        winner_king = board.king(winner)
        if not bishops or loser_king is None or winner_king is None:
            return 0.0

        bishop_sq = next(iter(bishops))
        bishop_light = chess.square_color(bishop_sq)
        corners = [chess.A1, chess.H1, chess.A8, chess.H8]
        target_corners = [c for c in corners if chess.square_color(c) == bishop_light]

        def corner_dist(corner: chess.Square) -> float:
            cr, cf = chess.square_rank(corner), chess.square_file(corner)
            kr, kf = chess.square_rank(loser_king), chess.square_file(loser_king)
            return abs(cr - kr) + abs(cf - kf)

        min_corner_dist = min(corner_dist(c) for c in target_corners)
        corner_bonus = max(0.0, 7.0 - min_corner_dist) * 18.0
        king_support = max(0.0, 14.0 - self._king_distance(winner_king, loser_king)) * 4.0
        signed = 130.0 + corner_bonus + king_support
        return signed if winner == chess.WHITE else -signed

    def _kpk_objective(self, board: chess.Board) -> float:
        white_score = self._pawn_race_score(board, chess.WHITE)
        black_score = self._pawn_race_score(board, chess.BLACK)
        return white_score - black_score

    def _pawn_race_score(self, board: chess.Board, color: chess.Color) -> float:
        pawns = board.pieces(chess.PAWN, color)
        if not pawns:
            return 0.0
        sign = 1.0 if color == chess.WHITE else -1.0
        total = 0.0
        for sq in pawns:
            rank = chess.square_rank(sq)
            advance = rank if color == chess.WHITE else 7 - rank
            total += 8.0 * advance

            if chess.square_rank(sq) in (5, 6) and color == chess.WHITE:
                total += 18.0
            if chess.square_rank(sq) in (1, 2) and color == chess.BLACK:
                total += 18.0
        return sign * total

    def _ladder_mate_objective(self, board: chess.Board) -> float:
        white_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
        black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))

        if self._is_lone_king(board, chess.BLACK) and white_rooks >= 2:
            return self._drive_king_score(chess.WHITE, chess.BLACK, board, base=150.0)
        if self._is_lone_king(board, chess.WHITE) and black_rooks >= 2:
            return self._drive_king_score(chess.BLACK, chess.WHITE, board, base=150.0)
        return 0.0
