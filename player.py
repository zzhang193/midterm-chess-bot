"""
=============================================================================
TransformerPlayer — Chess Player Submission
=============================================================================
A transformer-based chess player with a 3-tier failsafe system that
guarantees zero fallbacks (no illegal moves, no None outputs):

  Tier 1: Constrained Decoding (lm-format-enforcer)
           Forces the model to only generate tokens that form a legal UCI move.
           Expected success rate: ~99%

  Tier 2: Log-Probability Ranking
           Scores every legal move by its log-probability under the model.
           Picks the highest-scoring one. Mathematically guarantees a legal move.
           Only triggered if Tier 1 raises a Python exception.

  Tier 3: Return legal_moves[0]
           Absolute crash protection (e.g., OOM). Should never be reached.

Requirements: transformers, torch, chess, lm-format-enforcer
=============================================================================
"""

import chess
import torch
import re
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from chess_tournament import Player


class TransformerPlayer(Player):
    def __init__(self, name: str = "Ziyi_ChessBot"):
        super().__init__(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ──────────────────────────────────────────────────────────────
        self.model_name = "Ziyi193/chess-smollm2-135m"

        print(f"[{self.name}] Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Attempt to load lm-format-enforcer for constrained decoding
        self._has_enforcer = False
        try:
            from lmformatenforcer import RegexParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
            self._RegexParser = RegexParser
            self._build_prefix_fn = build_transformers_prefix_allowed_tokens_fn
            self._has_enforcer = True
            print(f"[{self.name}] Constrained decoding: ENABLED")
        except ImportError:
            print(f"[{self.name}] lm-format-enforcer not found; using log-prob fallback only")

    # ──────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────
    def get_move(self, fen: str) -> Optional[str]:
        """
        Given a FEN string, return the best legal UCI move.
        Uses a 3-tier failsafe to guarantee zero fallbacks.
        """
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        # No legal moves available (game over)
        if not legal_moves:
            return None

        # Optimization: if only one move is possible, play it immediately
        if len(legal_moves) == 1:
            return legal_moves[0]

        prompt = f"FEN: {fen} MOVE: "

        # Tier 1: Constrained Decoding — force output to be a legal move
        if self._has_enforcer:
            move = self._constrained_generate(prompt, legal_moves)
            if move:
                return move

        # Tier 2: Log-Probability Ranking — guaranteed legal move
        move = self._rank_by_logprob(prompt, legal_moves)
        if move:
            return move

        # Tier 3: Absolute crash protection
        return legal_moves[0]

    # ──────────────────────────────────────────────────────────────────
    # TIER 1: Constrained Decoding
    # ──────────────────────────────────────────────────────────────────
    def _constrained_generate(
        self, prompt: str, legal_moves: list[str]
    ) -> Optional[str]:
        """
        Use lm-format-enforcer to restrict the token space during generation
        so the model can ONLY produce sequences that match a legal UCI move.
        Builds a regex like (e2e4|d7d5|g1f3) and enforces it at each step.
        """
        try:
            # Escape special regex characters (e.g., promotion moves like e7e8q)
            escaped = [re.escape(m) for m in legal_moves]
            regex_pattern = "(" + "|".join(escaped) + ")"

            parser = self._RegexParser(regex_pattern)
            prefix_fn = self._build_prefix_fn(self.tokenizer, prompt, parser)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=6,  # UCI moves are at most 5 characters
                    prefix_allowed_tokens_fn=prefix_fn,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,   # Greedy: pick the top-scoring legal move
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            move_str = generated[len(prompt):].strip().split()[0]

            if move_str in legal_moves:
                return move_str

        except Exception:
            # If constrained decoding fails for any reason, fall through to Tier 2
            pass

        return None

    # ──────────────────────────────────────────────────────────────────
    # TIER 2: Log-Probability Ranking
    # ──────────────────────────────────────────────────────────────────
    def _rank_by_logprob(
        self, prompt: str, legal_moves: list[str]
    ) -> Optional[str]:
        """
        For each legal move, compute its log-probability under the model
        given the prompt. Return the move with the highest score.

        This is slower than generation (one forward pass per legal move)
        but GUARANTEES a legal move is returned every time.
        """
        try:
            best_move = None
            best_score = float("-inf")

            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )
            prompt_len = prompt_ids.shape[1]

            for move in legal_moves:
                # Encode the full string: prompt + candidate move
                full_text = prompt + move
                full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(
                    self.device
                )
                move_len = full_ids.shape[1] - prompt_len
                if move_len <= 0:
                    continue

                # Single forward pass to get logits for all positions
                with torch.no_grad():
                    logits = self.model(full_ids).logits  # (1, seq_len, vocab_size)

                # Sum the log-probabilities of each move token
                log_prob = 0.0
                for j in range(move_len):
                    # Position that predicts token at (prompt_len + j)
                    pos = prompt_len + j - 1
                    target = full_ids[0, prompt_len + j]
                    if 0 <= pos < logits.shape[1]:
                        probs = torch.log_softmax(logits[0, pos], dim=-1)
                        log_prob += probs[target].item()

                # Normalize by move length to avoid bias toward shorter moves
                score = log_prob / move_len

                if score > best_score:
                    best_score = score
                    best_move = move

            return best_move

        except Exception:
            # If even this fails (e.g., OOM), return first legal move
            return legal_moves[0] if legal_moves else None
