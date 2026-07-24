#!/usr/bin/env python3
"""CLI entrypoint for the play-harness watch/replay tool (spec §6).

Thin wrapper around ``game_theory_llm.play.viewer``. Examples::

    python3 scripts/watch_match.py LOG.jsonl                # static god-view replay
    python3 scripts/watch_match.py LOG.jsonl --follow        # tail a live match
    python3 scripts/watch_match.py LOG.jsonl --seat 2        # reconstruct seat 2's view
    python3 scripts/watch_match.py LOG.jsonl --god           # explicit god view (default)
    python3 scripts/watch_match.py LOG.jsonl --step          # event stepper (n/p/q)
    python3 scripts/watch_match.py LOG.jsonl --speed 4       # auto-play 4 events/sec
    python3 scripts/watch_match.py LOG.jsonl --html out.html # self-contained HTML replay
    python3 scripts/watch_match.py LOG.jsonl --metrics       # steering-metrics table

Equivalent to ``python3 -m game_theory_llm.play.viewer``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a bare script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from game_theory_llm.play.viewer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
