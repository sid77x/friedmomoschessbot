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

        def material_balance(side: chess.Color) -> int:
            own = 0
            opp = 0
            for sq, piece in board.piece_map().items():
                if piece.piece_type == chess.KING:
                    continue
                value = PIECE_ORDER.get(piece.piece_type, 0)
                if piece.color == side:
                    own += value
                else:
                    opp += value
            return own - opp

        def moved_piece_safety_penalty(move: chess.Move) -> int:
            mover = board.piece_at(move.from_square)
            if mover is None or mover.piece_type == chess.KING:
                return 0

            board.push(move)
            to_sq = move.to_square
            moved_piece = board.piece_at(to_sq)
            if moved_piece is None or moved_piece.piece_type == chess.KING:
                board.pop()
                return 0

            enemy = not moved_piece.color
            attacked = board.is_attacked_by(enemy, to_sq)
            defended = board.is_attacked_by(moved_piece.color, to_sq)

            penalty = 0
            if attacked and not defended:
                piece_value = PIECE_ORDER.get(moved_piece.piece_type, 0)
                penalty += int(piece_value * 2.2)
                if moved_piece.piece_type in (chess.QUEEN, chess.ROOK):
                    penalty += 250

            board.pop()
            return penalty

        def repetition_shuffle_penalty(move: chess.Move) -> int:
            mover = board.piece_at(move.from_square)
            if mover is None:
                return 0

            penalty = 0
            if len(board.move_stack) >= 2:
                last_own_move = board.move_stack[-2]
                if (
                    last_own_move.to_square == move.from_square
                    and last_own_move.from_square == move.to_square
                ):
                    penalty += 650

            if (
                board.fullmove_number >= 25
                and mover.piece_type in (chess.ROOK, chess.QUEEN)
                and not board.is_capture(move)
                and move.promotion is None
            ):
                penalty += 220

            return penalty

        def free_capture_bonus(move: chess.Move) -> int:
            if not board.is_capture(move):
                return 0
            victim = board.piece_at(move.to_square)
            if victim is None:
                return 0
            defended = board.is_attacked_by(victim.color, move.to_square)
            if not defended:
                return 500 + PIECE_ORDER.get(victim.piece_type, 0) // 3
            return 0

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
                total += free_capture_bonus(move)
            if move.promotion:
                total += 12_000 + PIECE_ORDER.get(move.promotion, 0)
            if move.uci() in killer_set:
                total += 5_000
            total += history_scores.get(move.uci(), 0)

            total -= moved_piece_safety_penalty(move)
            total -= repetition_shuffle_penalty(move)

            if material_balance(board.turn) > 350 and not board.is_capture(move):
                total -= 120

            board.push(move)
            if board.is_checkmate():
                total += 900_000
            elif board.is_stalemate():
                total -= 600_000
            if board.is_check():
                total += 700
            if move.promotion and move.promotion != chess.QUEEN and not board.is_checkmate():
                total -= 1800
            board.pop()
            return total

        return sorted(moves, key=score, reverse=True)
