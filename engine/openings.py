from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import chess


def position_key(board: chess.Board) -> str:
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    return f"{board.board_fen()} {board.turn} {board.castling_xfen()} {ep}"


OPENING_LINES_SAN: Dict[str, List[str]] = {
    "Ruy Lopez Main": ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6"],
    "Italian Giuoco Piano": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+"],
    "Italian Two Knights": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Na5", "Bb5+", "c6"],
    "Scotch Game": ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4", "Nf6", "Nxc6", "bxc6", "e5", "Qe7"],
    "Four Knights": ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6", "Bb5", "Bb4", "O-O", "O-O", "d3", "d6"],
    "Vienna Game": ["e4", "e5", "Nc3", "Nf6", "f4", "d5", "fxe5", "Nxe4", "Nf3", "Be7", "d3", "Nxc3"],
    "King's Gambit Accepted": ["e4", "e5", "f4", "exf4", "Nf3", "g5", "h4", "g4", "Ne5", "Nf6", "Bc4", "d5"],
    "Sicilian Najdorf": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be3", "e5"],
    "Sicilian Dragon": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6", "Be3", "Bg7"],
    "Sicilian Classical": ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "d6", "Be2", "e6"],
    "Sicilian Alapin": ["e4", "c5", "c3", "Nf6", "e5", "Nd5", "d4", "cxd4", "Nf3", "Nc6", "Bc4", "Nb6"],
    "French Winawer": ["e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5", "c5", "a3", "Bxc3+", "bxc3", "Ne7"],
    "French Tarrasch": ["e4", "e6", "d4", "d5", "Nd2", "Nf6", "e5", "Nfd7", "Bd3", "c5", "c3", "Nc6"],
    "French Advance": ["e4", "e6", "d4", "d5", "e5", "c5", "c3", "Nc6", "Nf3", "Qb6", "a3", "Nh6"],
    "Caro-Kann Classical": ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5", "Ng3", "Bg6", "h4", "h6"],
    "Caro-Kann Advance": ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nc3", "e6", "g4", "Bg6", "Nge2", "c5"],
    "Scandinavian Main": ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5", "d4", "c6", "Nf3", "Bf5", "Bc4", "e6"],
    "Pirc Defense": ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "Nf3", "Bg7", "Be2", "O-O", "O-O", "c6"],
    "Modern Defense": ["e4", "g6", "d4", "Bg7", "Nc3", "d6", "Nf3", "a6", "Be3", "b5", "Qd2", "Nd7"],
    "Alekhine Defense": ["e4", "Nf6", "e5", "Nd5", "d4", "d6", "Nf3", "Bg4", "Be2", "e6", "O-O", "Be7"],
    "Philidor Defense": ["e4", "e5", "Nf3", "d6", "d4", "Nf6", "Nc3", "Nbd7", "Bc4", "Be7", "O-O", "O-O"],
    "Queen's Gambit Declined": ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "h6"],
    "Queen's Gambit Accepted": ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6", "e3", "e6", "Bxc4", "c5", "O-O", "a6"],
    "Slav Defense": ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "dxc4", "a4", "Bf5", "e3", "e6"],
    "London System": ["d4", "d5", "Nf3", "Nf6", "Bf4", "e6", "e3", "c5", "c3", "Nc6", "Nbd2", "Bd6"],
    "Colle System": ["d4", "d5", "Nf3", "Nf6", "e3", "e6", "Bd3", "c5", "c3", "Nc6", "Nbd2", "Bd6"],
    "King's Indian Defense": ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O", "Be2", "e5"],
    "Grunfeld Defense": ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5", "e4", "Nxc3", "bxc3", "Bg7"],
    "Nimzo-Indian": ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O", "Bd3", "d5", "Nf3", "c5"],
    "Queen's Indian": ["d4", "Nf6", "c4", "e6", "Nf3", "b6", "g3", "Bb7", "Bg2", "Be7", "O-O", "O-O"],
    "Benoni Defense": ["d4", "Nf6", "c4", "c5", "d5", "e6", "Nc3", "exd5", "cxd5", "d6", "e4", "g6"],
    "Dutch Defense": ["d4", "f5", "g3", "Nf6", "Bg2", "g6", "Nf3", "Bg7", "O-O", "O-O", "c4", "d6"],
    "English Symmetrical": ["c4", "c5", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7", "e4", "d6", "Nge2", "e5"],
    "English Reversed Sicilian": ["c4", "e5", "Nc3", "Nf6", "g3", "d5", "cxd5", "Nxd5", "Bg2", "Nb6", "Nf3", "Be7"],
    "Catalan": ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O", "O-O", "dxc4"],
    "Bird Opening": ["f4", "d5", "Nf3", "Nf6", "e3", "g6", "b3", "Bg7", "Bb2", "O-O", "Be2", "c5"],
}


@dataclass
class OpeningBook:
    mapping: Dict[str, List[str]]

    @classmethod
    def from_builtin(cls) -> "OpeningBook":
        mapping: Dict[str, List[str]] = {}

        for line in OPENING_LINES_SAN.values():
            board = chess.Board()
            for san in line:
                key = position_key(board)
                try:
                    move = board.parse_san(san)
                except ValueError:
                    break
                mapping.setdefault(key, [])
                if move.uci() not in mapping[key]:
                    mapping[key].append(move.uci())
                board.push(move)

        return cls(mapping=mapping)

    def get_move(self, board: chess.Board) -> chess.Move | None:
        moves = self.mapping.get(position_key(board), [])
        if not moves:
            return None
        for uci in random.sample(moves, len(moves)):
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                return move
        return None
