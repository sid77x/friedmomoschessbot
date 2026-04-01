from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.D4,
    chess.E4,
    chess.F4,
    chess.C5,
    chess.D5,
    chess.E5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
]


@dataclass
class FeatureVector:
    values: List[float]
    names: List[str]


def _material(board: chess.Board) -> int:
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def _mobility(board: chess.Board) -> int:
    active = board.legal_moves.count()
    board.push(chess.Move.null())
    reply = board.legal_moves.count()
    board.pop()
    return active - reply


def _center_control(board: chess.Board) -> int:
    score = 0
    for sq in CENTER_SQUARES:
        score += len(board.attackers(chess.WHITE, sq))
        score -= len(board.attackers(chess.BLACK, sq))
    return score


def _space_advantage(board: chess.Board) -> int:
    score = 0
    for sq in EXTENDED_CENTER:
        if chess.square_rank(sq) >= 3 and board.is_attacked_by(chess.WHITE, sq):
            score += 1
        if chess.square_rank(sq) <= 4 and board.is_attacked_by(chess.BLACK, sq):
            score -= 1
    return score


def _pawn_structure(board: chess.Board) -> Dict[str, int]:
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    def count_doubled(pawns: chess.SquareSet) -> int:
        files = [0] * 8
        for sq in pawns:
            files[chess.square_file(sq)] += 1
        return sum(max(0, f - 1) for f in files)

    def count_isolated(pawns: chess.SquareSet) -> int:
        pawn_files = {chess.square_file(sq) for sq in pawns}
        isolated = 0
        for sq in pawns:
            f = chess.square_file(sq)
            if (f - 1 not in pawn_files) and (f + 1 not in pawn_files):
                isolated += 1
        return isolated

    def count_passed(pawns: chess.SquareSet, opp: chess.SquareSet, color: chess.Color) -> int:
        passed = 0
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            blocked = False
            files = [x for x in (f - 1, f, f + 1) if 0 <= x < 8]
            for of in files:
                for osq in opp:
                    if chess.square_file(osq) != of:
                        continue
                    orank = chess.square_rank(osq)
                    if color == chess.WHITE and orank > r:
                        blocked = True
                    if color == chess.BLACK and orank < r:
                        blocked = True
                if blocked:
                    break
            if not blocked:
                passed += 1
        return passed

    return {
        "doubled": count_doubled(white_pawns) - count_doubled(black_pawns),
        "isolated": count_isolated(white_pawns) - count_isolated(black_pawns),
        "passed": count_passed(white_pawns, black_pawns, chess.WHITE)
        - count_passed(black_pawns, white_pawns, chess.BLACK),
        "passed_advance": sum(chess.square_rank(sq) for sq in white_pawns)
        - sum(7 - chess.square_rank(sq) for sq in black_pawns),
    }


def _safe_king_moves(board: chess.Board, color: chess.Color) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    occupied_own = board.occupied_co[color]
    enemy = not color
    total = 0
    for sq in board.attacks(king_sq):
        if occupied_own & chess.BB_SQUARES[sq]:
            continue
        if board.is_attacked_by(enemy, sq):
            continue
        total += 1
    return total


def _king_safety(board: chess.Board) -> int:
    def pawn_shield(color: chess.Color) -> int:
        king_sq = board.king(color)
        if king_sq is None:
            return -500
        rank = chess.square_rank(king_sq)
        file_ = chess.square_file(king_sq)
        direction = 1 if color == chess.WHITE else -1
        shield_rank = rank + direction
        if not (0 <= shield_rank < 8):
            return 0
        total = 0
        for df in (-1, 0, 1):
            sf = file_ + df
            if not (0 <= sf < 8):
                continue
            sq = chess.square(sf, shield_rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                total += 1
        return total * 20

    return pawn_shield(chess.WHITE) - pawn_shield(chess.BLACK)


def _development(board: chess.Board) -> int:
    white_undeveloped = 0
    black_undeveloped = 0

    for sq in (chess.B1, chess.G1):
        piece = board.piece_at(sq)
        if piece and piece.color == chess.WHITE and piece.piece_type == chess.KNIGHT:
            white_undeveloped += 1
    for sq in (chess.C1, chess.F1):
        piece = board.piece_at(sq)
        if piece and piece.color == chess.WHITE and piece.piece_type == chess.BISHOP:
            white_undeveloped += 1

    for sq in (chess.B8, chess.G8):
        piece = board.piece_at(sq)
        if piece and piece.color == chess.BLACK and piece.piece_type == chess.KNIGHT:
            black_undeveloped += 1
    for sq in (chess.C8, chess.F8):
        piece = board.piece_at(sq)
        if piece and piece.color == chess.BLACK and piece.piece_type == chess.BISHOP:
            black_undeveloped += 1

    return black_undeveloped - white_undeveloped


def _early_queen_penalty(board: chess.Board) -> int:
    if board.fullmove_number > 12:
        return 0

    penalty_white = 0
    penalty_black = 0

    wq = board.pieces(chess.QUEEN, chess.WHITE)
    bq = board.pieces(chess.QUEEN, chess.BLACK)
    if wq and chess.D1 not in wq and _development(board) < 2:
        penalty_white = 1
    if bq and chess.D8 not in bq and _development(board) > -2:
        penalty_black = 1

    return penalty_white - penalty_black


def _king_activity(board: chess.Board) -> int:
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 0

    def dist_to_center(sq: chess.Square) -> int:
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        return min(abs(r - 3), abs(r - 4)) + min(abs(f - 3), abs(f - 4))

    return dist_to_center(bk) - dist_to_center(wk)


def _king_confinement(board: chess.Board) -> int:
    white_king_mob = _safe_king_moves(board, chess.WHITE)
    black_king_mob = _safe_king_moves(board, chess.BLACK)
    return (8 - black_king_mob) - (8 - white_king_mob)


def _hanging_material(board: chess.Board) -> int:
    score = 0

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        enemy = not piece.color
        attacked = board.is_attacked_by(enemy, sq)
        defended = board.is_attacked_by(piece.color, sq)
        if attacked and not defended:
            penalty = PIECE_VALUES.get(piece.piece_type, 0)
            score += -penalty if piece.color == chess.WHITE else penalty
    return score


def _tactical_pressure(board: chess.Board) -> int:
    score = 0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        attackers_white = len(board.attackers(chess.WHITE, sq))
        attackers_black = len(board.attackers(chess.BLACK, sq))
        weight = PIECE_VALUES.get(piece.piece_type, 0) // 100
        if piece.color == chess.BLACK:
            score += attackers_white * weight
            score -= attackers_black * weight
        else:
            score -= attackers_black * weight
            score += attackers_white * weight
    return score


def game_phase(board: chess.Board) -> str:
    non_pawn = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        non_pawn += len(board.pieces(piece_type, chess.WHITE))
        non_pawn += len(board.pieces(piece_type, chess.BLACK))

    if board.fullmove_number <= 10 and non_pawn >= 12:
        return "opening"
    if non_pawn <= 6:
        return "endgame"
    return "midgame"


def extract_feature_dict(board: chess.Board) -> Dict[str, float]:
    pawn_data = _pawn_structure(board)
    return {
        "material": float(_material(board)),
        "mobility": float(_mobility(board)),
        "center_control": float(_center_control(board)),
        "space": float(_space_advantage(board)),
        "king_safety": float(_king_safety(board)),
        "development": float(_development(board)),
        "early_queen_penalty": float(_early_queen_penalty(board)),
        "king_activity": float(_king_activity(board)),
        "king_confinement": float(_king_confinement(board)),
        "hanging_material": float(_hanging_material(board)),
        "tactical_pressure": float(_tactical_pressure(board)),
        "pawn_doubled": float(pawn_data["doubled"]),
        "pawn_isolated": float(pawn_data["isolated"]),
        "pawn_passed": float(pawn_data["passed"]),
        "pawn_passed_advance": float(pawn_data["passed_advance"]),
    }


def extract_feature_vector(board: chess.Board) -> FeatureVector:
    feature_map = extract_feature_dict(board)
    names = list(feature_map.keys())
    values = [feature_map[n] for n in names]
    turn = 1.0 if board.turn == chess.WHITE else -1.0
    values.append(turn)
    names.append("side_to_move")
    return FeatureVector(values=values, names=names)
