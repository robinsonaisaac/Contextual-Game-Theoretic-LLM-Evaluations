"""Tier-5 shared configuration: single source of truth for token-budget constants.

Import MAX_TOKEN_CEILING from here in build_tier5_ledger.py and tier5_gate.py instead of
hardcoding a ceiling in each — those two disagreed (8192 vs 12000) before this module
existed. scripts/tier5_evalsuite.sh cannot import Python; its inline --max-tokens literals
(12000 for long in-domain/extrapolated cells, smaller values for shorter benchmarks) must be
kept in sync with MAX_TOKEN_CEILING by hand — see the comment at its top.
"""
from __future__ import annotations

# Ceiling for the longest ("extrapolated"/long-cell) eval and build rows. Gold canonical
# traces at the largest trained/extrapolated horizons run ~3.4k-7.4k tokens, and the
# naturalized (wordier) SFT completions need headroom beyond that; 4096 CENSORED these
# cells in an earlier run (truncation before the ANSWER tail -> parse_rate 0), it did not
# measure them. 12000 is the largest ceiling any Tier-5 tool currently needs.
MAX_TOKEN_CEILING = 12000
