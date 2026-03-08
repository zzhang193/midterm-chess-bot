# Chess Transformer Player

## Project Overview

This project fine-tunes `SmolLM2-135M-Instruct` to play chess by predicting the best UCI move given a FEN board state. The inference system uses a 3-tier architecture that guarantees zero fallback errors during tournament gameplay.

**Model:** [Ziyi193/chess-smollm2-135m](https://huggingface.co/Ziyi193/chess-smollm2-135m)

## Training Data

The training dataset combines two sources to maximize both quality and robustness:

1. **Lichess Tactical Puzzles (~50,000 samples):** Streamed from the HuggingFace `lichess/chess-puzzles` dataset in streaming mode to avoid Colab storage limits. Filtered for puzzle rating >= 1500 to ensure high-quality tactical positions covering forks, pins, skewers, and forced checkmates.

2. **Chaotic Positions + Stockfish (~3,000 samples):** Random moves are played from the starting position to create messy, out-of-distribution board states. Stockfish then provides the best recovery move. This addresses the exposure bias problem: without these samples, the model would only know "clean" positions and collapse when facing unpredictable opponents.

## Fine-Tuning Details

- **Base model:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Method:** LoRA (r=16, alpha=32) on attention layers, merged into base weights before pushing to HuggingFace
- **Prompt format:** `FEN: <fen_string> MOVE: <uci_move>`
- **Epochs:** 3 (eval_loss reached ~1.12 by epoch 2; early stopping via `load_best_model_at_end=True` prevented overfitting)
- **Training environment:** Google Colab Free Tier (T4 GPU)

To handle Colab session instability, checkpoints were saved to a mounted Google Drive path. This successfully recovered training progress after a mid-training session disconnect. Logging and evaluation frequency were also reduced (`logging_steps=500`, `eval_steps=1000`) to prevent browser memory issues during long training runs.

## Inference Architecture: 3-Tier Zero-Fallback System

The `player.py` submission uses a layered failsafe to guarantee that every call to `get_move()` returns a legal move:

**Tier 1 — Constrained Decoding:**
Uses `lm-format-enforcer` to restrict the token space during generation. A regex pattern is built from all current legal moves (e.g., `(e2e4|d7d5|g1f3)`), so the model can only produce valid UCI strings. Expected success rate: ~99%.

**Tier 2 — Log-Probability Ranking:**
If Tier 1 raises an exception, the system computes the log-probability of every legal move under the model and returns the highest-scoring one. This is slower (one forward pass per legal move) but mathematically guarantees a legal output.

**Tier 3 — Absolute Failsafe:**
Returns `legal_moves[0]` if the model encounters an unrecoverable error (e.g., out-of-memory). Should never be reached in practice.

## Reproducibility

### Requirements

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes: `transformers`, `torch`, `chess`, `lm-format-enforcer`.

### Local Testing

```python
from player import TransformerPlayer
from chess_tournament import Game, RandomPlayer

bot = TransformerPlayer("Ziyi_ChessBot")

for i in range(5):
    game = Game(bot, RandomPlayer("Random"), max_half_moves=200)
    print(f"Game {i+1}: {game.play()}")
```

### Repository Structure

```
├── player.py           # Submission: TransformerPlayer class
├── requirements.txt    # All dependencies for the grading server
└── README.md           # This file
```
