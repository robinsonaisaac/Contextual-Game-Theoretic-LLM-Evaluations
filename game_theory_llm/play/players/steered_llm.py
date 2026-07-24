"""Activation-steered LLM player backed by a Modal `SteeringWorker`.

This player dispatches each `act()` call to the deployed `safety` Modal
app, which holds a long-lived `SteeringWorker` instance for the chosen
(model_name, layer, position, alpha) combination. The worker applies the
steering hook for the duration of one generation and returns the parsed
trace.

For now we reuse the existing `eval_shard` interface (which expects a
list of "stories"). We wrap each game prompt as a single-story list of
length 1 with story_id derived from the match log path. This is
suboptimal (per-call container amortisation is poor) but it works
end-to-end without a new Modal entrypoint. A follow-up `act_one()`
method on the worker would batch better.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import modal

from ..base import Game


class SteeredLLMPlayer:
    """Modal-backed activation-steered LLM player."""

    def __init__(
        self,
        *,
        model_short: str,        # registry short name, e.g. "E4B"
        hf_id: str,              # e.g. "google/gemma-4-E4B-it"
        gpu_tier: str,           # "small" or "large"
        run_id: str,             # /data/runs/{run_id}/vectors.pt must exist
        layer: int,
        position: str = "mean_trace",
        alpha: float = 0.0,
        name: Optional[str] = None,
        app_name: str = "safety",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        max_history: int = 60,
    ):
        self.model_short = model_short
        self.hf_id = hf_id
        self.gpu_tier = gpu_tier
        self.run_id = run_id
        self.layer = layer
        self.position = position
        self.alpha = alpha
        self.name = name or f"Steered[{model_short},L{layer}/{position},α={alpha:+.1f}]"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_history = max_history
        self.history: list[dict] = []
        cls_name = "SteeringWorker" if gpu_tier == "small" else "SteeringWorkerLarge"
        Cls = modal.Cls.from_name(app_name, cls_name)
        self._worker = Cls(model_name=hf_id)

    def act(self, game: Game, state: Any, player_idx: int) -> str:
        prompt = game.render_prompt(state, player_idx)
        full_prompt = self._build_prompt(prompt)
        # Cast as a one-story batch for eval_shard reuse.
        story = {
            "story_id": f"match__{game.name}__seat{player_idx}__t{len(self.history)}",
            "prompt": full_prompt,
            "coop_choice": "A",     # unused at α=0 except by parse path; required field
            "seed": int.from_bytes(os.urandom(4), "big"),
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "apply_chat_template": True,
        }
        # Use eval_shard.remote (blocking) — slower than batched but
        # adequate for v0. The result writes a shard parquet on the
        # volume; we discard it (the trace text is what we want, which
        # we recover via the eval_progress.jsonl side-channel).
        result = self._worker.eval_shard.remote(
            run_id=self.run_id, layer=self.layer, position=self.position,
            alpha=float(self.alpha), stories=[story],
            result_subdir=f"play/{game.name}",
        )
        # Pull the actual trace text from the per-story progress side-channel.
        text = self._fetch_progress_trace(story["story_id"])
        # Bookkeeping.
        self._append("game", prompt)
        self._append(self.name, text)
        return text

    def receive_observation(self, obs: dict) -> None:
        t = obs.get("type")
        if t == "action":
            self._append("public", f"player {obs['player']} -> {obs['action']}")
        elif t == "parse_error":
            self._append("public",
                         f"SYSTEM: your previous reply could not be parsed "
                         f"({obs['error']}). Reply EXACTLY in the required "
                         f"format.")
        elif t == "phase_change":
            self._append("public", f"phase change: {obs.get('to')}")
        elif t == "message":
            frm = obs.get("from")
            text = obs.get("text", "")
            if obs.get("scope") == "private":
                to = ",".join(str(x) for x in obs.get("to", []))
                self._append("public", f"whisper P{frm}->[{to}]: {text}")
            else:
                self._append("public", f"P{frm} (public): {text}")
        elif t == "message_meta":
            self._append("public",
                         f"P{obs.get('from')} whispered to "
                         f"{obs.get('n_recipients')} player(s)")
        elif t == "alliance_event":
            self._append("public",
                         f"alliance #{obs.get('alliance_id')} "
                         f"{obs.get('event')} by P{obs.get('actor')} "
                         f"(members {obs.get('members')})")

    def _append(self, who: str, text: str) -> None:
        self.history.append({"who": who, "text": text})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _build_prompt(self, current: str) -> str:
        parts = []
        if self.history:
            parts.append("=== Game history ===")
            for h in self.history:
                parts.append(f"[{h['who']}] {h['text']}")
            parts.append("=== End history ===\n")
        parts.append(current)
        return "\n".join(parts)

    def _fetch_progress_trace(self, story_id: str) -> str:
        """Read back the steered trace from the eval_progress JSONL.

        Best-effort: if we can't get it, return empty text so the runner
        falls back to the first legal action.
        """
        import json
        import subprocess
        local = Path(f"local_data/play_progress_{self.run_id}.jsonl")
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.call(
            ["python3", "-m", "modal", "volume", "get", "--force",
             "safety", f"runs/{self.run_id}/eval_progress.jsonl", str(local)],
            stderr=subprocess.DEVNULL,
        )
        if not local.exists():
            return ""
        # Scan from the bottom for the matching story_id.
        for line in reversed(local.read_text().splitlines()):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("story_id") == story_id:
                return r.get("trace", "")
        return ""
