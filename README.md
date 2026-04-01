# Ensemble Chess Engine (AI Lab Project)

This engine is a modular Python chess engine designed for GUI play, UCI usage, and Lichess bot integration.

It combines:

- Classical heuristic evaluation
- ML evaluator (RandomForest fallback to linear model)
- Neural evaluator (PyTorch when available)
- Positional evaluator
- RL TD evaluator
- RL deep value evaluator
- Phase-aware weighted ensemble

## Recent Upgrades Included

The following upgrades are implemented in this codebase:

1. Stability and time-management improvements

- UCI think-time bounding to prevent long stalls
- Better correspondence responsiveness
- Improved engine fallback behavior so a legal move is still produced when needed
- Faster correspondence re-check cycle support via config (smaller checkin period)

2. Search and tactical behavior improvements

- Forced mate-in-1 detection at root
- Draw/repetition handling that avoids draw loops when clearly winning
- Stalemate-avoidance penalty when ahead
- Better quiescence handling for checking lines

3. Tactical feature upgrades

- Explicit pin pressure feature
- Explicit skewer pressure feature
- Explicit fork pressure feature
- Tactical features integrated into heuristic and positional evaluators

4. Opening-principle discipline upgrades

- Added castling completion feature for opening objectives
- Added early-rook-move penalty feature to discourage premature rook lifts
- Increased opening weight on development + castling, and stronger penalties for early queen/rook drift

5. Endgame and checkmate curriculum

- Endgame mating objectives in heuristic evaluator:
  - KQK
  - KRK
  - KBNK
  - KPK
  - Ladder mate motifs
- Training curriculum module with:
  - One-queen mate drills
  - One-rook mate drills
  - Single-pawn conversion drills
  - Ladder mate drills
  - Knight+bishop mating-net drills
  - Generated mate-in-1 and mate-in-2 tactical samples
  - Non-mate tactical/strategic samples (development, castling, defensive play, anti-early-rook drift)
  - Exchange and sacrifice motif samples
  - Decoy / deflection / discovered-attack themes
  - Fork, double-attack, and skewer themes
  - Promotion-priority samples (favoring queen promotion over weak underpromotion)

6. RL integration and training

- Added RL TD model: models/rl_td_model.pkl
- Added RL deep model: models/rl_deep_model.pt (requires torch)
- Integrated RL models into ensemble phase weights
- Added readiness gating: untrained neural/RL models do not inject random noise
- Added ensemble weight renormalization over available trained models only

7. Unified all-model training pipeline

- scripts/train_models.py now trains all available models in one run:
  - ML
  - Neural (if torch installed)
  - RL TD
  - RL deep (if torch installed)
- ML training is incremental across sessions when the feature schema is unchanged
  (RandomForest warm-start adds trees instead of resetting the model)
- Uses mixed random-position data plus mixed curriculum data (mates + tactics + principles)
- Supports fixed time budget with --minutes
- Saves models in chess engine/models regardless of launch directory

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
    positional.py
    ml_model.py
    neural_model.py
    rl_td_model.py
    rl_deep_model.py
    checkmate_curriculum.py
    ensemble.py
main.py
scripts/train_models.py
models/
requirements.txt
README.md

## Setup

1. Create and activate environment

- Windows PowerShell:

  .\.venv\Scripts\Activate.ps1

2. Install dependencies

  pip install -r requirements.txt

3. Install torch (required for neural + deep RL)

  pip install torch --index-url https://download.pytorch.org/whl/cpu

If torch is unavailable, the engine still runs and trains ML + RL TD + heuristic + positional components.

## Training (Recommended)

Run unified training from repository root:

  .\chess engine\.venv\Scripts\python.exe .\chess engine\scripts\train_models.py --minutes 18

Optional flags:

- --minutes 15
- --skip-neural
- --skip-rl
- --ml-samples 8000 --nn-samples 10000 --nn-epochs 12

Train with master-level PGN supervision (recommended for stronger middlegame plans):

  .\chess engine\.venv\Scripts\python.exe .\chess engine\scripts\train_models.py --minutes 18 --pgn ".\data\masters.pgn" --min-elo 2200 --max-pgn-games 1500 --pgn-sample-every 4

PGN notes:

- Use strong game datasets (Lichess elite games, titled events, or curated master PGNs).
- The trainer blends game-result supervision with heuristic labels.
- PGN positions are added on top of random positions and mixed curriculum, not as a replacement.

Expected artifacts:

- chess engine/models/ml_model.pkl
- chess engine/models/neural_model.pt (if torch available)
- chess engine/models/rl_td_model.pkl
- chess engine/models/rl_deep_model.pt (if torch available)

## Run Modes

GUI mode:

  python -m engine.main --mode gui --depth 4 --think-time 1.8

UCI mode:

  python -m engine.main --mode uci

Lichess API mode (direct):

  python -m engine.main --mode lichess --token YOUR_TOKEN

## Lichess-Bot Integration (Recommended)

Use lichess-bot to run this engine in production.

From lichess-bot root:

  .\chess engine\.venv\Scripts\python.exe .\lichess-bot.py --config .\config.yml

Engine protocol support includes:

- uci, isready, ucinewgame, quit
- position startpos|fen ... moves ...
- go movetime/depth/wtime/btime/winc/binc/movestogo
- setoption Skill Level and Move Overhead

## Notes on Model Readiness

- Ensemble uses readiness-aware weights.
- If a model file is not present (or torch is unavailable), that model gets weight 0 and remaining model weights are renormalized.
- This keeps evaluation stable and avoids random untrained outputs.
- If feature dimensions change (for example after adding new features), models with incompatible old checkpoints are safely reinitialized or partially loaded.

## Security and Publishing

- Do not commit secrets (tokens, .env, private keys).
- Keep config.yml private; use config.yml.default and examples for sharing.
- .gitignore in repo root is hardened for venvs, local models, logs, and sensitive files.
