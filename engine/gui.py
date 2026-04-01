from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

import chess

from .search import SearchEngine


PIECE_GLYPHS = {
    "P": "\u2659",
    "N": "\u2658",
    "B": "\u2657",
    "R": "\u2656",
    "Q": "\u2655",
    "K": "\u2654",
    "p": "\u265F",
    "n": "\u265E",
    "b": "\u265D",
    "r": "\u265C",
    "q": "\u265B",
    "k": "\u265A",
}

PIECE_FILES = {
    "P": "wp.png",
    "N": "wn.png",
    "B": "wb.png",
    "R": "wr.png",
    "Q": "wq.png",
    "K": "wk.png",
    "p": "bp.png",
    "n": "bn.png",
    "b": "bb.png",
    "r": "br.png",
    "q": "bq.png",
    "k": "bk.png",
}


class ChessGUI:
    def __init__(self, engine: SearchEngine, depth: int = 4, think_time_s: float = 1.8) -> None:
        self.engine = engine
        self.depth = depth
        self.think_time_s = think_time_s

        self.board = chess.Board()
        self.selected_square: int | None = None
        self.last_move: chess.Move | None = None
        self.ai_thinking = False
        self.animating = False

        self.root = tk.Tk()
        self.root.title("Ensemble Chess Engine")

        self.square_size = 72
        self.canvas_size = self.square_size * 8

        self.mode_var = tk.StringVar(value="human-vs-ai")
        self.human_color_var = tk.StringVar(value="white")
        self.eval_text = tk.StringVar(value="Eval: 0.00")
        self.status_text = tk.StringVar(value="Ready")
        self.piece_images: dict[str, tk.PhotoImage] = {}

        self._load_piece_images()

        self._build_ui()
        self._draw_board()
        self._update_labels()

        self.root.after(150, self._tick)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Mode:").pack(side=tk.LEFT)
        mode_menu = ttk.OptionMenu(top, self.mode_var, self.mode_var.get(), "human-vs-ai", "ai-vs-ai")
        mode_menu.pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Human:").pack(side=tk.LEFT)
        color_menu = ttk.OptionMenu(top, self.human_color_var, self.human_color_var.get(), "white", "black")
        color_menu.pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="New Game", command=self._new_game).pack(side=tk.LEFT, padx=8)

        ttk.Label(top, textvariable=self.eval_text).pack(side=tk.RIGHT, padx=10)

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_click)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(bottom, textvariable=self.status_text).pack(side=tk.LEFT)

    def run(self) -> None:
        self.root.mainloop()

    def _new_game(self) -> None:
        self.board = chess.Board()
        self.selected_square = None
        self.last_move = None
        self.ai_thinking = False
        self.animating = False
        self.status_text.set("New game started")
        self._draw_board()
        self._update_labels()

    def _on_click(self, event: tk.Event) -> None:
        if self.ai_thinking or self.animating or self.board.is_game_over(claim_draw=True):
            return
        if not self._is_human_turn():
            return

        file_ = int(event.x // self.square_size)
        rank = 7 - int(event.y // self.square_size)
        if not (0 <= file_ < 8 and 0 <= rank < 8):
            return

        sq = chess.square(file_, rank)
        piece = self.board.piece_at(sq)

        if self.selected_square is None:
            if piece and piece.color == self.board.turn:
                self.selected_square = sq
                self._draw_board()
            return

        move = chess.Move(self.selected_square, sq)
        legal = list(self.board.legal_moves)
        if move not in legal:
            move = self._promotion_or_none(self.selected_square, sq, legal)

        if move is not None and move in legal:
            self._animate_and_commit_move(move)
        else:
            if piece and piece.color == self.board.turn:
                self.selected_square = sq
            else:
                self.selected_square = None
            self._draw_board()

    def _promotion_or_none(
        self, from_sq: int, to_sq: int, legal_moves: list[chess.Move]
    ) -> chess.Move | None:
        for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            candidate = chess.Move(from_sq, to_sq, promotion=promo)
            if candidate in legal_moves:
                return candidate
        return None

    def _load_piece_images(self) -> None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        pieces_dir = os.path.join(project_root, "assets", "pieces")

        for symbol, filename in PIECE_FILES.items():
            path = os.path.join(pieces_dir, filename)
            if os.path.exists(path):
                try:
                    self.piece_images[symbol] = tk.PhotoImage(file=path)
                except tk.TclError:
                    continue

    def _square_center(self, square: int) -> tuple[float, float]:
        file_ = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file_ * self.square_size + self.square_size / 2
        y = (7 - rank) * self.square_size + self.square_size / 2
        return x, y

    def _captured_square(self, move: chess.Move) -> int | None:
        if self.board.is_en_passant(move):
            return move.to_square - 8 if self.board.turn == chess.WHITE else move.to_square + 8
        if self.board.is_capture(move):
            return move.to_square
        return None

    def _render_piece(self, piece: chess.Piece, x: float, y: float) -> None:
        image = self.piece_images.get(piece.symbol())
        if image is not None:
            self.canvas.create_image(x, y, image=image)
            return

        glyph = PIECE_GLYPHS.get(piece.symbol(), piece.symbol())
        self.canvas.create_text(
            x,
            y,
            text=glyph,
            font=("Segoe UI Symbol", 42),
            fill="#111111" if piece.color == chess.BLACK else "#ffffff",
        )

    def _draw_board(
        self,
        hidden_squares: set[int] | None = None,
        moving_piece: tuple[chess.Piece, float, float] | None = None,
    ) -> None:
        self.canvas.delete("all")
        hidden_squares = hidden_squares or set()
        light = "#f0d9b5"
        dark = "#b58863"
        sel = "#85c1e9"
        hint = "#82e0aa"

        for rank in range(8):
            for file_ in range(8):
                x1 = file_ * self.square_size
                y1 = rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                sq = chess.square(file_, 7 - rank)
                color = light if (file_ + rank) % 2 == 0 else dark

                if self.last_move and (sq == self.last_move.from_square or sq == self.last_move.to_square):
                    color = hint
                if self.selected_square == sq:
                    color = sel

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                if sq in hidden_squares:
                    continue

                piece = self.board.piece_at(sq)
                if piece:
                    self._render_piece(piece, (x1 + x2) / 2, (y1 + y2) / 2)

        if moving_piece is not None:
            piece, x, y = moving_piece
            self._render_piece(piece, x, y)

    def _animate_and_commit_move(
        self,
        move: chess.Move,
        after_commit: Callable[[], None] | None = None,
    ) -> None:
        moving_piece = self.board.piece_at(move.from_square)
        if moving_piece is None:
            return

        self.animating = True
        capture_square = self._captured_square(move)
        hidden_squares = {move.from_square}
        if capture_square is not None:
            hidden_squares.add(capture_square)

        start_x, start_y = self._square_center(move.from_square)
        end_x, end_y = self._square_center(move.to_square)

        steps = 12
        duration_ms = 160
        frame_ms = max(1, duration_ms // steps)

        def frame(step: int) -> None:
            t = step / steps
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t
            self._draw_board(hidden_squares=hidden_squares, moving_piece=(moving_piece, x, y))

            if step < steps:
                self.root.after(frame_ms, lambda: frame(step + 1))
                return

            self.board.push(move)
            self.last_move = move
            self.selected_square = None
            self.animating = False
            self._draw_board()
            self._update_labels()
            if after_commit is not None:
                after_commit()

        frame(0)

    def _is_human_turn(self) -> bool:
        if self.mode_var.get() == "ai-vs-ai":
            return False
        human_white = self.human_color_var.get().lower() == "white"
        return (self.board.turn == chess.WHITE and human_white) or (
            self.board.turn == chess.BLACK and not human_white
        )

    def _tick(self) -> None:
        if not self.ai_thinking and not self.animating and not self.board.is_game_over(claim_draw=True):
            if self.mode_var.get() == "ai-vs-ai" or not self._is_human_turn():
                self._start_ai_move()

        self._update_labels()
        self.root.after(150, self._tick)

    def _start_ai_move(self) -> None:
        self.ai_thinking = True
        self.status_text.set("AI thinking...")
        board_snapshot = self.board.copy(stack=False)

        def worker() -> None:
            try:
                info = self.engine.choose_move(
                    board_snapshot,
                    max_depth=self.depth,
                    time_limit_s=self.think_time_s,
                )
            except Exception as exc:
                def apply_error() -> None:
                    self.ai_thinking = False
                    self.status_text.set(f"AI error: {exc}")

                self.root.after(0, apply_error)
                return

            def apply_move() -> None:
                if info.best_move is None:
                    self.ai_thinking = False
                    self.status_text.set("No legal moves")
                    return
                if info.best_move in self.board.legal_moves:
                    def done() -> None:
                        self.ai_thinking = False
                        self.status_text.set(
                            f"AI: {info.best_move.uci()} depth={info.depth_reached} "
                            f"score={info.score:.1f} nodes={info.nodes} time={info.elapsed_ms:.0f}ms"
                        )

                    self._animate_and_commit_move(info.best_move, after_commit=done)
                    return

                self.ai_thinking = False
                self.status_text.set("Engine produced illegal move")

            self.root.after(0, apply_move)

        threading.Thread(target=worker, daemon=True).start()

    def _update_labels(self) -> None:
        eval_stm = self.engine.evaluator.evaluate(self.board)
        eval_white = eval_stm if self.board.turn == chess.WHITE else -eval_stm
        self.eval_text.set(f"Eval: {eval_white / 100.0:+.2f}")

        if self.board.is_game_over(claim_draw=True):
            self.status_text.set(f"Game over: {self.board.result(claim_draw=True)}")
