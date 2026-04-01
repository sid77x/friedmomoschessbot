# Ensemble Chess Engine (AI Lab Project)

A modular production-style chess engine in Python that combines multiple AI paradigms:

- Classical heuristic evaluation
- Lightweight ML evaluation (RandomForest fallback to linear model)
- Lightweight neural evaluation (PyTorch, optional)
- Positional activity model
- Weighted ensemble with dynamic game-phase blending

Supports two primary modes:

1. Local GUI mode (human vs AI, AI vs AI)
2. Online mode through Lichess Bot API

Also includes optional UCI mode for Arena/CuteChess.

## Project Structure

engine/
  board.py
  move_generator.py
  search.py
  openings.py
  lichess_bot.py
  gui.py
  uci.py
  main.py
  evaluation/
      features.py
      heuristic.py
      ml_model.py
      neural_model.py
      positional.py
      ensemble.py
main.py
scripts/train_models.py
requirements.txt
README.md

## Key Features

- Full chess rules via python-chess bitboard engine:
  - legal move generation
  - castling
  - en passant
  - promotions
  - game state and move history
- Search engine:
  - Minimax/Negamax with Alpha-Beta pruning
  - Iterative deepening
  - Move ordering (TT move, captures, killers, history)
  - Quiescence search
  - Transposition table
- Ensemble evaluation:
  - final_score = w1*M1 + w2*M2 + w3*M3 + w4*M4
  - Dynamic phase-aware weights (opening/midgame/endgame)
- Opening book:
  - 35+ popular opening lines
  - roughly 6 to 7 moves deep per line
  - instant move selection when a match exists
- Logging/search stats:
  - nodes searched
  - quiescence nodes
  - time per move
  - depth reached
  - model score breakdown support

## Setup

1) Create and activate a Python environment.

2) Install dependencies:

pip install -r requirements.txt

3) Optional neural model dependency:

pip install torch

If torch is not installed, the engine still runs with heuristic + ML + positional evaluators.

## Train ML and Neural Components

Quick training (recommended for project demo):

python -m engine.main --mode train

Custom training:

python -m engine.main --mode train --ml-samples 6000 --nn-samples 8000 --nn-epochs 10

Skip neural training:

python -m engine.main --mode train --skip-neural

Artifacts are saved to:

- models/ml_model.pkl
- models/neural_model.pt

Training data is generated from randomized legal positions and labeled by the classical evaluator for lightweight CPU training.

## Run GUI Mode

Human vs AI (default):

python -m engine.main --mode gui --depth 4 --think-time 1.8

Switch to AI vs AI in the GUI dropdown.

GUI shows:

- board
- last move highlighting
- live evaluation
- move/depth/nodes/time status

Piece visuals:

- Place piece PNG files in assets/pieces using names:
  wp.png wn.png wb.png wr.png wq.png wk.png bp.png bn.png bb.png br.png bq.png bk.png
- If these files are present, the GUI uses image sprites.
- If not present, it falls back to Unicode chess glyphs (not letter symbols).

## Run Lichess Bot Mode

1) Create a Lichess bot account and token.
2) Set token as environment variable or pass it directly.

Using environment variable:

set LICHESS_BOT_TOKEN=YOUR_TOKEN_HERE
python -m engine.main --mode lichess --depth 5 --think-time 2.0

Using CLI argument:

python -m engine.main --mode lichess --token YOUR_TOKEN_HERE

Behavior:

- Listens to incoming events
- Accepts standard challenges
- Streams game state
- Selects and sends moves via engine
- Basic resign/draw offering logic
- Handles time control with adaptive think time

## Recommended Lichess Integration (lichess-bot)

The most reliable way to run this engine on Lichess is through lichess-bot using UCI mode.

1) Install and set up lichess-bot from:
https://github.com/lichess-bot-devs/lichess-bot

2) Copy this project template into your lichess-bot folder as config.yml:

- lichess-bot-config.yml.example

3) Update token and paths in config.yml, then run lichess-bot.

Engine command used by lichess-bot:

python -m engine.main --mode uci

This engine now supports the UCI commands typically used by lichess-bot:

- uci / isready / ucinewgame / quit
- position startpos|fen ... moves ...
- go with movetime/depth and standard clock args (wtime, btime, winc, binc, movestogo)
- setoption for Skill Level and Move Overhead

Security note:

- If you posted or exposed your Lichess token anywhere public, rotate/regenerate it in your Lichess account before use.

## Optional UCI Mode

python -m engine.main --mode uci

Then connect engine process in Arena/CuteChess as a UCI engine.

## Performance Notes

- CPU-friendly search target: depth 3 to 5
- Uses transposition table and move ordering for speed
- Book moves are instant

## Suggested Demo Flow (AI Lab)

1) Train models quickly on laptop CPU
2) Show Human vs AI GUI game
3) Switch to AI vs AI to show autonomous play
4) Explain ensemble breakdown and phase-weight changes
5) Show Lichess bot integration architecture

## Limitations

- Synthetic training labels are heuristic-based, not super-GM quality
- No opening learning from online games yet
- Endgame tablebases are not included

## Next Improvements

- Add persisted transposition table cache between moves/games
- Add PGN ingestion for stronger supervised labels
- Add stronger time management and contempt settings
- Expand draw/resign heuristics with material-aware logic
