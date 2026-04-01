from __future__ import annotations

import json
import threading
import time
from typing import Dict, Generator, Optional

import chess
import requests

from .search import SearchEngine


class LichessBot:
    def __init__(
        self,
        token: str,
        engine: SearchEngine,
        base_url: str = "https://lichess.org",
    ) -> None:
        self.token = token
        self.engine = engine
        self.base_url = base_url.rstrip("/")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/x-ndjson",
                "User-Agent": "ensemble-chess-engine/1.0",
            }
        )

        self.username = self._fetch_username().lower()
        self.bad_eval_counts: Dict[str, int] = {}

    def _fetch_username(self) -> str:
        resp = self.session.get(f"{self.base_url}/api/account", timeout=20)
        resp.raise_for_status()
        return resp.json().get("id", "")

    def run(self) -> None:
        print(f"Logged in as: {self.username}")
        while True:
            try:
                for event in self._stream_json("/api/stream/event"):
                    etype = event.get("type")
                    if etype == "challenge":
                        self._handle_challenge(event.get("challenge", {}))
                    elif etype == "gameStart":
                        game_id = event.get("game", {}).get("id")
                        if game_id:
                            t = threading.Thread(target=self.play_game, args=(game_id,), daemon=True)
                            t.start()
            except requests.RequestException as exc:
                print(f"Event stream error: {exc}. Reconnecting...")
                time.sleep(2.0)

    def _handle_challenge(self, challenge: Dict) -> None:
        challenge_id = challenge.get("id")
        if not challenge_id:
            return

        variant = challenge.get("variant", {}).get("key", "")
        speed = challenge.get("speed", "")
        if variant != "standard" or speed == "ultraBullet":
            self._post(f"/api/challenge/{challenge_id}/decline", data={"reason": "standard"})
            return

        self._post(f"/api/challenge/{challenge_id}/accept")
        print(f"Accepted challenge: {challenge_id}")

    def play_game(self, game_id: str) -> None:
        print(f"Game started: {game_id}")
        initial_board = chess.Board()
        my_color: Optional[chess.Color] = None

        for event in self._stream_json(f"/api/bot/game/stream/{game_id}"):
            etype = event.get("type")
            if etype == "gameFull":
                initial_fen = event.get("initialFen", "startpos")
                initial_board = chess.Board() if initial_fen == "startpos" else chess.Board(initial_fen)

                white_id = event.get("white", {}).get("id", "").lower()
                black_id = event.get("black", {}).get("id", "").lower()
                if white_id == self.username:
                    my_color = chess.WHITE
                elif black_id == self.username:
                    my_color = chess.BLACK

                state = event.get("state", {})
                board = self._build_current_board(initial_board, state.get("moves", ""))
                if my_color is not None:
                    self._maybe_play_move(game_id, board, my_color, state)

            elif etype == "gameState":
                status = event.get("status", "started")
                if status != "started":
                    print(f"Game {game_id} ended with status: {status}")
                    return

                if my_color is None:
                    continue

                board = self._build_current_board(initial_board, event.get("moves", ""))
                self._maybe_play_move(game_id, board, my_color, event)

    def _build_current_board(self, initial: chess.Board, moves_blob: str) -> chess.Board:
        board = initial.copy(stack=False)
        if not moves_blob:
            return board

        for uci in moves_blob.split():
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                board.push(mv)
            else:
                break
        return board

    def _maybe_play_move(
        self,
        game_id: str,
        board: chess.Board,
        my_color: chess.Color,
        state: Dict,
    ) -> None:
        if board.turn != my_color or board.is_game_over(claim_draw=True):
            return

        remaining_ms = state.get("wtime", 30000) if my_color == chess.WHITE else state.get("btime", 30000)
        increment_ms = state.get("winc", 0) if my_color == chess.WHITE else state.get("binc", 0)

        think_time = self._allocate_time(remaining_ms, increment_ms)
        info = self.engine.choose_move(board, max_depth=5, time_limit_s=think_time)

        if info.best_move is None:
            self._post(f"/api/bot/game/{game_id}/resign")
            return

        score = info.score
        self._update_resign_state(game_id, score)
        if self.bad_eval_counts.get(game_id, 0) >= 3 and remaining_ms < 15000:
            print(f"Resigning lost game {game_id}")
            self._post(f"/api/bot/game/{game_id}/resign")
            return

        offer_draw = abs(score) < 25 and board.fullmove_number > 45
        move_endpoint = f"/api/bot/game/{game_id}/move/{info.best_move.uci()}"
        params = {"offeringDraw": "true"} if offer_draw else None
        self._post(move_endpoint, params=params)

        print(
            f"{game_id} move={info.best_move.uci()} depth={info.depth_reached} "
            f"score={info.score:.1f} nodes={info.nodes} t={info.elapsed_ms:.0f}ms"
        )

    def _allocate_time(self, remaining_ms: int, increment_ms: int) -> float:
        remaining_s = max(remaining_ms, 0) / 1000.0
        increment_s = max(increment_ms, 0) / 1000.0

        target = remaining_s * 0.04 + increment_s * 0.6
        if remaining_s < 15.0:
            target = min(target, 1.2)
        return max(0.2, min(target, 8.0))

    def _update_resign_state(self, game_id: str, score: float) -> None:
        if score < -1200:
            self.bad_eval_counts[game_id] = self.bad_eval_counts.get(game_id, 0) + 1
        else:
            self.bad_eval_counts[game_id] = 0

    def _stream_json(self, path: str) -> Generator[Dict, None, None]:
        with self.session.get(f"{self.base_url}{path}", stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue

    def _post(self, path: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> None:
        resp = self.session.post(f"{self.base_url}{path}", data=data, params=params, timeout=20)
        if resp.status_code >= 400:
            print(f"POST {path} failed: {resp.status_code} {resp.text[:120]}")
