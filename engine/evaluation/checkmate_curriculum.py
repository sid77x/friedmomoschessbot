from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import chess


@dataclass
class CurriculumPosition:
    board: chess.Board
    target_cp: float
    tag: str


def _cp_for(color: chess.Color, value: float) -> float:
    return value if color == chess.WHITE else -value


def _from_fen(fen: str, target_cp: float, tag: str) -> CurriculumPosition:
    return CurriculumPosition(board=chess.Board(fen), target_cp=target_cp, tag=tag)


def base_checkmate_curriculum() -> List[CurriculumPosition]:
    # Static drills for common mating patterns and winning endgames.
    return [
        _from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1", 90000.0, "mate_in_1_queen"),
        _from_fen("6k1/5R2/6K1/8/8/8/8/8 w - - 0 1", 86000.0, "mate_in_1_rook"),
        _from_fen("6k1/6R1/6R1/6K1/8/8/8/8 w - - 0 1", 86000.0, "ladder_mate"),
        _from_fen("7k/8/8/3K4/8/8/3BN3/8 w - - 0 1", 380.0, "knight_bishop_mate_net"),
        _from_fen("4k3/4P3/4K3/8/8/8/8/8 w - - 0 1", 260.0, "single_pawn_promotion"),
        _from_fen("7K/5Q2/7k/8/8/8/8/8 b - - 0 1", -90000.0, "defend_vs_mate_in_1"),
    ]


def _candidate_moves(board: chess.Board, max_branch: int = 16) -> List[chess.Move]:
    legal = list(board.legal_moves)
    checks: List[chess.Move] = []
    captures: List[chess.Move] = []
    others: List[chess.Move] = []

    for move in legal:
        if board.is_capture(move):
            captures.append(move)
            continue
        if board.gives_check(move):
            checks.append(move)
            continue
        if move.promotion is not None:
            checks.append(move)
            continue
        others.append(move)

    ordered = checks + captures + others
    return ordered[:max_branch]


def _can_force_mate(board: chess.Board, attacker: chess.Color, plies: int) -> bool:
    if board.is_checkmate():
        return board.turn != attacker
    if board.is_stalemate() or board.is_insufficient_material():
        return False
    if plies <= 0:
        return False

    moves = _candidate_moves(board)
    if not moves:
        return False

    if board.turn == attacker:
        for move in moves:
            board.push(move)
            forced = _can_force_mate(board, attacker, plies - 1)
            board.pop()
            if forced:
                return True
        return False

    for move in moves:
        board.push(move)
        forced = _can_force_mate(board, attacker, plies - 1)
        board.pop()
        if not forced:
            return False
    return True


def _place(board: chess.Board, piece_type: chess.PieceType, color: chess.Color, rng: random.Random) -> bool:
    squares = list(chess.SQUARES)
    rng.shuffle(squares)
    for sq in squares:
        if board.piece_at(sq) is not None:
            continue
        board.set_piece_at(sq, chess.Piece(piece_type, color))
        if board.is_valid():
            return True
        board.remove_piece_at(sq)
    return False


def _random_mini_endgame(rng: random.Random) -> chess.Board:
    b = chess.Board(None)

    king_squares = list(chess.SQUARES)
    rng.shuffle(king_squares)
    wk = king_squares[0]
    bk = None
    for sq in king_squares[1:]:
        if chess.square_distance(wk, sq) > 1:
            bk = sq
            break
    if bk is None:
        bk = chess.E8

    b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

    attacker = rng.choice([chess.WHITE, chess.BLACK])
    defender = not attacker

    attacker_templates: List[Dict[chess.PieceType, int]] = [
        {chess.QUEEN: 1},
        {chess.ROOK: 1},
        {chess.ROOK: 2},
        {chess.BISHOP: 1, chess.KNIGHT: 1},
        {chess.PAWN: 1},
        {chess.QUEEN: 1, chess.PAWN: 1},
        {chess.ROOK: 1, chess.PAWN: 1},
    ]
    defender_templates: List[Dict[chess.PieceType, int]] = [
        {},
        {chess.PAWN: 1},
        {chess.PAWN: 2},
        {chess.ROOK: 1},
    ]

    for piece_type, count in rng.choice(attacker_templates).items():
        for _ in range(count):
            _place(b, piece_type, attacker, rng)

    for piece_type, count in rng.choice(defender_templates).items():
        for _ in range(count):
            _place(b, piece_type, defender, rng)

    b.turn = attacker
    if not b.is_valid() or b.is_game_over(claim_draw=True):
        return _random_mini_endgame(rng)
    return b


def generated_mate_curriculum(
    mate_in: int,
    max_samples: int = 10,
    seed: int = 7,
    max_attempts: int = 1600,
) -> List[CurriculumPosition]:
    rng = random.Random(seed + mate_in * 101)
    samples: List[CurriculumPosition] = []
    required_plies = 2 * mate_in - 1

    attempts = 0
    while attempts < max_attempts and len(samples) < max_samples:
        attempts += 1
        board = _random_mini_endgame(rng)
        attacker = board.turn

        if not _can_force_mate(board, attacker, required_plies):
            continue
        if required_plies > 1 and _can_force_mate(board, attacker, required_plies - 2):
            continue

        target = _cp_for(attacker, 90000.0 - 80.0 * mate_in)
        samples.append(CurriculumPosition(board=board.copy(stack=True), target_cp=target, tag=f"mate_in_{mate_in}"))

    return samples


def build_training_curriculum(seed: int = 7) -> List[CurriculumPosition]:
    curriculum = base_checkmate_curriculum()
    curriculum.extend(generated_mate_curriculum(mate_in=1, max_samples=10, seed=seed))
    curriculum.extend(generated_mate_curriculum(mate_in=2, max_samples=12, seed=seed + 1))
    return curriculum
