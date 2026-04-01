from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import chess

from .evaluation.ensemble import EnsembleBreakdown, EnsembleEvaluator
from .move_generator import MoveGenerator
from .openings import OpeningBook, position_key


INF = 10**9
MATE_SCORE = 100000


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: str
    best_move: Optional[chess.Move]


@dataclass
class SearchInfo:
    best_move: Optional[chess.Move]
    score: float
    depth_reached: int
    elapsed_ms: float
    nodes: int
    qnodes: int
    tt_hits: int
    source: str
    breakdown: Optional[EnsembleBreakdown]


class SearchEngine:
    def __init__(
        self,
        evaluator: Optional[EnsembleEvaluator] = None,
        opening_book: Optional[OpeningBook] = None,
        max_quiescence_depth: int = 4,
    ) -> None:
        self.evaluator = evaluator or EnsembleEvaluator()
        self.opening_book = opening_book or OpeningBook.from_builtin()
        self.move_generator = MoveGenerator()
        self.max_quiescence_depth = max_quiescence_depth

        self.tt: Dict[str, TTEntry] = {}
        self.killers: Dict[int, Tuple[Optional[chess.Move], Optional[chess.Move]]] = {}
        self.history_scores: Dict[str, int] = {}
        self.eval_cache: Dict[str, float] = {}

        self.nodes = 0
        self.qnodes = 0
        self.tt_hits = 0

        self._time_deadline = math.inf
        self._timed_out = False

    def choose_move(
        self,
        board: chess.Board,
        max_depth: int = 4,
        time_limit_s: float = 2.0,
    ) -> SearchInfo:
        start = time.perf_counter()
        self.nodes = 0
        self.qnodes = 0
        self.tt_hits = 0
        self.eval_cache.clear()
        self._timed_out = False
        self._time_deadline = start + max(time_limit_s, 0.01)

        book_move = self.opening_book.get_move(board)
        if book_move is not None:
            score, breakdown = self.evaluator.evaluate(board, return_breakdown=True)
            elapsed = (time.perf_counter() - start) * 1000.0
            return SearchInfo(
                best_move=book_move,
                score=score,
                depth_reached=0,
                elapsed_ms=elapsed,
                nodes=0,
                qnodes=0,
                tt_hits=0,
                source="opening_book",
                breakdown=breakdown,
            )

        for move in list(board.legal_moves):
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                elapsed = (time.perf_counter() - start) * 1000.0
                _, breakdown = self.evaluator.evaluate(board, return_breakdown=True)
                return SearchInfo(
                    best_move=move,
                    score=MATE_SCORE - 1,
                    depth_reached=1,
                    elapsed_ms=elapsed,
                    nodes=0,
                    qnodes=0,
                    tt_hits=0,
                    source="forced_mate",
                    breakdown=breakdown,
                )

        best_move: Optional[chess.Move] = None
        best_score = -INF
        depth_reached = 0

        for depth in range(1, max_depth + 1):
            if self._out_of_time():
                break

            score, move = self._search_root(board, depth)
            if self._timed_out:
                break

            if move is not None:
                best_move = move
                best_score = score
                depth_reached = depth

        if best_move is None:
            legal = list(board.legal_moves)
            if legal:
                best_move = legal[0]
                best_score, breakdown = self.evaluator.evaluate(board, return_breakdown=True)
            else:
                best_score, breakdown = 0.0, None
        else:
            _, breakdown = self.evaluator.evaluate(board, return_breakdown=True)

        elapsed = (time.perf_counter() - start) * 1000.0
        return SearchInfo(
            best_move=best_move,
            score=best_score,
            depth_reached=depth_reached,
            elapsed_ms=elapsed,
            nodes=self.nodes,
            qnodes=self.qnodes,
            tt_hits=self.tt_hits,
            source="search",
            breakdown=breakdown,
        )

    def _search_root(self, board: chess.Board, depth: int) -> Tuple[float, Optional[chess.Move]]:
        alpha = -INF
        beta = INF
        best_score = -INF
        best_move = None
        static_eval = self._static_eval(board)

        key = position_key(board)
        tt_move = self.tt.get(key).best_move if key in self.tt else None
        moves = self.move_generator.ordered_moves(
            board,
            list(board.legal_moves),
            tt_move=tt_move,
            killers=self.killers.get(0, (None, None)),
            history_scores=self.history_scores,
        )

        for move in moves:
            if self._out_of_time():
                self._timed_out = True
                break

            board.push(move)
            if board.is_checkmate():
                score = MATE_SCORE - 1
            elif board.is_stalemate() and static_eval > 150:
                # Avoid giving away wins by forcing stalemate from a winning position.
                score = -200.0
            else:
                score = -self._alpha_beta(board, depth - 1, -beta, -alpha, ply=1)
            board.pop()

            if self._timed_out:
                break

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        if best_move is not None:
            self.tt[key] = TTEntry(depth=depth, score=best_score, flag="EXACT", best_move=best_move)

        return best_score, best_move

    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        if self._out_of_time():
            self._timed_out = True
            return 0.0

        self.nodes += 1

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        if board.is_repetition(2) or board.can_claim_draw():
            return self._draw_score(board)

        if depth <= 0:
            if board.is_check():
                depth = 1
            else:
                return self._quiescence(board, alpha, beta, qdepth=0)

        key = position_key(board)
        old_alpha = alpha

        tt_entry = self.tt.get(key)
        if tt_entry and tt_entry.depth >= depth:
            self.tt_hits += 1
            if tt_entry.flag == "EXACT":
                return tt_entry.score
            if tt_entry.flag == "LOWER":
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == "UPPER":
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score

        tt_move = tt_entry.best_move if tt_entry else None
        killers = self.killers.get(ply, (None, None))
        moves = self.move_generator.ordered_moves(
            board,
            list(board.legal_moves),
            tt_move=tt_move,
            killers=killers,
            history_scores=self.history_scores,
        )

        best_move = None
        best_score = -INF

        for move in moves:
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if self._timed_out:
                return 0.0

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if not board.is_capture(move):
                    self._store_killer(ply, move)
                    self.history_scores[move.uci()] = self.history_scores.get(move.uci(), 0) + depth * depth
                break

        if best_move is None:
            return -MATE_SCORE + ply

        flag = "EXACT"
        if best_score <= old_alpha:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"
        self.tt[key] = TTEntry(depth=depth, score=best_score, flag=flag, best_move=best_move)

        return best_score

    def _quiescence(self, board: chess.Board, alpha: float, beta: float, qdepth: int) -> float:
        if self._out_of_time():
            self._timed_out = True
            return 0.0

        self.qnodes += 1

        stand_pat = self._static_eval(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if qdepth >= self.max_quiescence_depth:
            return alpha

        tactical_moves = [m for m in board.legal_moves if board.is_capture(m)]
        if board.is_check():
            tactical_moves = list(board.legal_moves)
        else:
            for move in board.legal_moves:
                if board.is_capture(move):
                    continue
                if self._is_checking_move(board, move):
                    tactical_moves.append(move)

        ordered = self.move_generator.ordered_moves(board, tactical_moves)

        for move in ordered:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, qdepth + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _store_killer(self, ply: int, move: chess.Move) -> None:
        k1, k2 = self.killers.get(ply, (None, None))
        if k1 is None or k1 != move:
            self.killers[ply] = (move, k1)
        else:
            self.killers[ply] = (k1, k2)

    def _out_of_time(self) -> bool:
        return time.perf_counter() >= self._time_deadline

    def _is_checking_move(self, board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        is_check = board.is_check()
        board.pop()
        return is_check

    def _draw_score(self, board: chess.Board) -> float:
        static = self._static_eval(board)
        if static > 180:
            return -80.0
        if static < -180:
            return 80.0
        return 0.0

    def _static_eval(self, board: chess.Board) -> float:
        key = position_key(board)
        if key in self.eval_cache:
            return self.eval_cache[key]

        # Fast node evaluation for search speed: avoid expensive ML/RL calls on
        # every interior node. Full ensemble is still used for final breakdown.
        value: float
        heuristic_eval = getattr(self.evaluator, "heuristic", None)
        positional_eval = getattr(self.evaluator, "positional_model", None)

        if heuristic_eval is not None and positional_eval is not None:
            h = float(heuristic_eval.evaluate(board))
            p = float(positional_eval.evaluate(board))
            value = 0.72 * h + 0.28 * p
        elif heuristic_eval is not None:
            value = float(heuristic_eval.evaluate(board))
        else:
            value = float(self.evaluator.evaluate(board))

        self.eval_cache[key] = value
        return value
