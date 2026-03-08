"""
=============================================================================
TransformerPlayer — Chess Player Submission
=============================================================================
A transformer-based chess player with a 4-tier system that combines neural
network evaluation with chess heuristics for stronger play:

  Tier 0: Instant Tactics (no model calls)
           Checkmate detection + tactical override of model blunders.
           Catches free pieces the model might miss or give away.

  Tier 1: Constrained Decoding (lm-format-enforcer)
           Forces the model to only generate tokens that form a legal UCI move.
           After generation, a tactical override checks for obvious blunders.

  Tier 2: Log-Probability Ranking + Chess Heuristics
           Scores every legal move by model log-prob + heuristic bonus
           (captures, checks, hanging-piece penalty, promotion, center control).
           Mathematically guarantees a legal move.

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

        # Piece values for chess heuristics (MVV-LVA capture ordering)
        self._piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

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

        # Tier 0: Instant tactical checks (no model calls)
        # — Always play checkmate if available
        for m in board.legal_moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m.uci()
            board.pop()

        prompt = f"FEN: {fen} MOVE: "

        # Tier 1: Constrained Decoding — force output to be a legal move
        if self._has_enforcer:
            move = self._constrained_generate(prompt, legal_moves)
            if move:
                # Tactical override: if the model's choice hangs a piece
                # and there's an obvious good capture, prefer the capture
                override = self._tactical_override(board, move, legal_moves)
                return override if override else move

        # Tier 2: Log-Probability Ranking + Heuristics — guaranteed legal move
        move = self._rank_by_logprob(prompt, legal_moves, board)
        if move:
            return move

        # Tier 3: Absolute crash protection
        return legal_moves[0]

    # ──────────────────────────────────────────────────────────────────
    # TIER 0: Tactical Override (fast, no model calls)
    # ──────────────────────────────────────────────────────────────────
    def _tactical_override(
        self, board: chess.Board, model_move: str, legal_moves: list[str]
    ) -> Optional[str]:
        """
        Quick check: if the model's chosen move hangs a valuable piece,
        and there is an obvious safe capture available, override the model.
        Returns None if the model's move looks fine.
        """
        try:
            model_chess_move = chess.Move.from_uci(model_move)

            # Check if model's move puts our piece on an attacked square
            # without a worthwhile trade
            model_penalty = 0
            dest = model_chess_move.to_square
            moving_piece = board.piece_at(model_chess_move.from_square)
            if moving_piece and board.is_attacked_by(not board.turn, dest):
                our_val = self._piece_values.get(moving_piece.piece_type, 0)
                if board.is_capture(model_chess_move):
                    captured = board.piece_at(dest)
                    cap_val = self._piece_values.get(captured.piece_type, 0) if captured else 0
                    model_penalty = max(0, our_val - cap_val)
                else:
                    model_penalty = our_val

            # If model's move doesn't hang anything significant, trust it
            if model_penalty < 3:  # Only override for knight/bishop/rook/queen hanging
                return None

            # Look for the best safe capture as alternative
            best_alt = None
            best_gain = 0
            for uci in legal_moves:
                m = chess.Move.from_uci(uci)
                if not board.is_capture(m):
                    continue
                captured = board.piece_at(m.to_square)
                cap_val = self._piece_values.get(captured.piece_type, 0) if captured else 0
                attacker = board.piece_at(m.from_square)
                att_val = self._piece_values.get(attacker.piece_type, 0) if attacker else 0

                # Net gain: captured value minus risk if square is defended
                if board.is_attacked_by(not board.turn, m.to_square):
                    net = cap_val - att_val
                else:
                    net = cap_val

                if net > best_gain:
                    best_gain = net
                    best_alt = uci

            return best_alt  # None if no good capture found

        except Exception:
            return None

    def _compute_heuristic(self, board: chess.Board, uci: str) -> float:
        """
        Compute a chess heuristic bonus for a candidate move.
        Returns a bonus on a scale calibrated to normalized log-probs (~±5).
        Board state is BEFORE the move is played.
        """
        try:
            move = chess.Move.from_uci(uci)
            bonus = 0.0

            # --- Check / checkmate after move ---
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return 100.0
            if board.is_check():
                bonus += 0.5
            board.pop()

            # --- Capture bonus (MVV-LVA: Most Valuable Victim, Least Valuable Attacker) ---
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    cap_val = self._piece_values.get(captured.piece_type, 0)
                    attacker = board.piece_at(move.from_square)
                    att_val = self._piece_values.get(attacker.piece_type, 0) if attacker else 1
                    # Scale: capturing a queen with a pawn → 9/1 * 1.5 = 13.5
                    bonus += (cap_val / max(att_val, 1)) * 1.5
                elif board.is_en_passant(move):
                    bonus += 1.5

            # --- Promotion bonus ---
            if move.promotion:
                bonus += self._piece_values.get(move.promotion, 0) * 1.0

            # --- Penalty for moving piece to attacked square (hanging) ---
            if board.is_attacked_by(not board.turn, move.to_square):
                moving_piece = board.piece_at(move.from_square)
                if moving_piece:
                    piece_val = self._piece_values.get(moving_piece.piece_type, 0)
                    if board.is_capture(move):
                        captured = board.piece_at(move.to_square)
                        cap_val = self._piece_values.get(captured.piece_type, 0) if captured else 0
                        if piece_val > cap_val:
                            bonus -= (piece_val - cap_val) * 0.8
                    else:
                        bonus -= piece_val * 0.8

            # --- Center control (minor) ---
            if move.to_square in {chess.E4, chess.D4, chess.E5, chess.D5}:
                bonus += 0.3

            return bonus

        except Exception:
            return 0.0

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
        self, prompt: str, legal_moves: list[str], board: chess.Board
    ) -> Optional[str]:
        """
        For each legal move, compute its log-probability under the model
        given the prompt, plus a chess heuristic bonus. Return the move
        with the highest combined score.

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
                # Then add chess heuristic bonus
                score = (log_prob / move_len) + self._compute_heuristic(board, move)

                if score > best_score:
                    best_score = score
                    best_move = move

            return best_move

        except Exception:
            # If even this fails (e.g., OOM), return first legal move
            return legal_moves[0] if legal_moves else None
