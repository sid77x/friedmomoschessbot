from __future__ import annotations

import sys

import chess

from .search import SearchEngine


class UCIProtocol:
    def __init__(self, engine: SearchEngine) -> None:
        self.engine = engine
        self.board = chess.Board()
        self.default_depth = 5
        self.default_movetime_ms = 1800
        self.max_think_time_s = 20.0

    def run(self) -> None:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line == "uci":
                print("id name EnsembleChessEngine")
                print("id author AI Lab")
                print("option name Skill Level type spin default 15 min 0 max 20")
                print("option name Move Overhead type spin default 100 min 0 max 2000")
                print("uciok")
            elif line == "isready":
                print("readyok")
            elif line.startswith("setoption"):
                self._handle_setoption(line)
            elif line.startswith("position"):
                self._handle_position(line)
            elif line.startswith("go"):
                self._handle_go(line)
            elif line == "stop":
                # Search is synchronous in this lightweight engine.
                pass
            elif line == "ucinewgame":
                self.board = chess.Board()
                self.engine.tt.clear()
            elif line == "quit":
                break
            sys.stdout.flush()

    def _handle_setoption(self, line: str) -> None:
        # UCI requires accepting options; in this project we parse known names and ignore others.
        tokens = line.split()
        lowered = [t.lower() for t in tokens]
        if "name" not in lowered:
            return

        name_idx = lowered.index("name") + 1
        if name_idx >= len(tokens):
            return

        value = None
        if "value" in lowered:
            value_idx = lowered.index("value") + 1
            if value_idx < len(tokens):
                value = " ".join(tokens[value_idx:])

        name_tokens = tokens[name_idx : lowered.index("value")] if "value" in lowered else tokens[name_idx:]
        option_name = " ".join(name_tokens).lower()

        if option_name == "skill level" and value is not None:
            try:
                level = max(0, min(20, int(value)))
                self.default_depth = max(2, min(8, 2 + level // 3))
            except ValueError:
                return
        elif option_name == "move overhead" and value is not None:
            try:
                self.default_movetime_ms = max(150, 1800 - min(1200, int(value)))
            except ValueError:
                return

    def _handle_position(self, line: str) -> None:
        tokens = line.split()
        if "startpos" in tokens:
            self.board = chess.Board()
            if "moves" in tokens:
                idx = tokens.index("moves") + 1
                for uci in tokens[idx:]:
                    self.board.push(chess.Move.from_uci(uci))
            return

        if "fen" in tokens:
            fen_idx = tokens.index("fen") + 1
            if "moves" in tokens:
                moves_idx = tokens.index("moves")
                fen = " ".join(tokens[fen_idx:moves_idx])
                moves = tokens[moves_idx + 1 :]
            else:
                fen = " ".join(tokens[fen_idx:])
                moves = []
            self.board = chess.Board(fen)
            for uci in moves:
                self.board.push(chess.Move.from_uci(uci))

    def _handle_go(self, line: str) -> None:
        tokens = line.split()
        lower = [t.lower() for t in tokens]

        depth = self.default_depth
        movetime_ms = self.default_movetime_ms

        if "depth" in lower:
            idx = lower.index("depth") + 1
            if idx < len(tokens):
                depth = max(1, int(tokens[idx]))

        if "movetime" in lower:
            idx = lower.index("movetime") + 1
            if idx < len(tokens):
                movetime_ms = max(50, int(tokens[idx]))
        else:
            wtime = self._read_int_param(tokens, lower, "wtime")
            btime = self._read_int_param(tokens, lower, "btime")
            winc = self._read_int_param(tokens, lower, "winc", default=0)
            binc = self._read_int_param(tokens, lower, "binc", default=0)
            movestogo = self._read_int_param(tokens, lower, "movestogo", default=30)
            if wtime is not None and btime is not None:
                remaining = wtime if self.board.turn == chess.WHITE else btime
                increment = winc if self.board.turn == chess.WHITE else binc
                movetime_ms = self._allocate_time_ms(remaining, increment, movestogo)

        requested_time_s = max(0.05, movetime_ms / 1000.0)
        bounded_time_s = min(requested_time_s, self.max_think_time_s)

        info = self.engine.choose_move(
            self.board.copy(stack=True),
            max_depth=depth,
            time_limit_s=bounded_time_s,
        )

        elapsed = max(1, int(info.elapsed_ms))
        nps = int((info.nodes * 1000) / elapsed) if info.nodes > 0 else 0
        cp = int(info.score)
        print(
            f"info depth {info.depth_reached} score cp {cp} nodes {info.nodes} "
            f"time {elapsed} nps {nps} pv {info.best_move.uci() if info.best_move else '0000'}"
        )
        if info.best_move is None:
            print("bestmove 0000")
        else:
            print(f"bestmove {info.best_move.uci()}")

    def _read_int_param(
        self,
        tokens: list[str],
        lower: list[str],
        key: str,
        default: int | None = None,
    ) -> int | None:
        if key not in lower:
            return default
        idx = lower.index(key) + 1
        if idx >= len(tokens):
            return default
        try:
            return int(tokens[idx])
        except ValueError:
            return default

    def _allocate_time_ms(self, remaining_ms: int, increment_ms: int, movestogo: int) -> int:
        moves_left = max(8, movestogo)
        base = remaining_ms // moves_left
        inc_bonus = int(increment_ms * 0.7)
        budget = base + inc_bonus
        safety_cap = max(150, int(remaining_ms * 0.25))
        return max(80, min(budget, safety_cap))
