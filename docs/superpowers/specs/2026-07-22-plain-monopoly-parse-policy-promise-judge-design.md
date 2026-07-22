# Plain Monopoly + Parse-Void Policy + Promise Judge — Design

**Date:** 2026-07-22
**Approved by:** Isaac Robinson (design walkthrough + 4 scope answers in-session)
**Repo:** steering worktree (`.worktrees/steering`), branch `feature/activation-steering`

## Goal

Fix the cooperation-measurement problems found by the 2026-07-22 audit
(`docs/results/cooperation_measurement_audit.md`): remove the silent
random-move fallback, delete Diplomacy, rebuild Monopoly as the plain game
with a real trade dialogue, and measure cooperation from complete game-aware
transcripts plus an LLM promise/reneging ledger — across all four remaining
games.

## User decisions (binding)

1. **Diplomacy: delete the code** (engine, tests, map, all references).
   Historical run data and result docs stay untouched.
2. **Parse failures: retry up to 3 times, then void the match** and record
   which model/seat failed and why ("might mean we steered too much").
   Cancellation rate is a headline per-condition metric. No random-move
   substitution anywhere.
3. **Monopoly: plain game.** No negotiation rounds, no whispers, no alliance
   pacts. All talk lives in the end-of-turn trade dialogue.
4. **Promise judge: all four games now** (Monopoly, ONW, Secret Hitler, Risk).

## Workstream A — Delete Diplomacy

Delete outright:
- `game_theory_llm/play/games/diplomacy_lite.py`
- `game_theory_llm/play/maps/diplomacy_map.py`
- `tests/play/test_diplomacy.py`, `tests/play/test_maps_diplomacy.py`

Update every live reference (grep-verified: `games/__init__.py`,
`maps/__init__.py` if it exports, `play/viewer.py` board art,
`play/messaging.py` + `play/players/llm.py` docstring mentions,
`play/games/risk_lite.py` comparative comments, `steering/modal_app.py`
worker registry, `scripts/play_steering_experiment.py` game choices,
`scripts/analyze_game_steering.py` `COOP_SIDE`,
`tests/test_play_harness.py` parametrization). While touching the registry,
add `MonopolyLite` to `games/__init__.py` (currently missing).

`data/runs/dip_steering_v1/` and `docs/results/*` are historical records —
not modified.

## Workstream B — Parse policy (shared runner, all games)

In `game_theory_llm/play/runner.py::run_match`:

- `max_parse_retries` default 2 → **3** (4 total attempts).
- The existing retry loop stays (it already logs `parse_error` records and
  delivers a `{"type": "parse_error", "error": ...}` observation to the
  player before re-calling `act()`).
- **Requirement on both LLM player classes** (`players/llm.py`,
  `players/steered_llm.py`): a `parse_error` observation must render into the
  next prompt as an explicit corrective line, e.g.
  `SYSTEM: your previous reply could not be parsed (<error>). Reply EXACTLY
  in the required format.` Verify/implement in both classes.
- **Replace the fallback block** (`runner.py:151-159`): when retries are
  exhausted, log
  ```json
  {"type": "aborted", "reason": "unparseable_output", "turn": N,
   "player": seat, "model": <player .model_key or .name>,
   "steering": <steering_tag>, "phase": <state.phase>,
   "last_error": "...", "last_raw": "<=600 chars"}
  ```
  then stop the match loop. **No `terminal` record is written** for an
  aborted match. `MatchResult.metadata` gains
  `aborted=True, aborted_player, aborted_reason`. All-zero rewards.
- The `fallback` record type and `rng.choice(legal)` substitution are
  removed entirely. `RandomPlayer` is unaffected (returns pre-parsed dicts).
- `max_turns` safety cap behavior unchanged.

Analysis contract: a log containing an `aborted` record is excluded from
every behavior metric; per-condition `cancel_rate` = aborted / total matches
is reported in `aggregate.json` and the console table, and each aborted
per-match row carries `aborted=True, aborted_player, model, steering`.

## Workstream C — Plain Monopoly (rewrite of `monopoly_lite.py`)

Same file, same class name `MonopolyLite`, same board/cash/dice constants —
but **no `MessagingMixin`/`AllianceMixin`**, no negotiation phase, no pacts,
no heuristic trade-candidate menu shown to models.

### Turn structure

1. **Auto-resolve roll** (`_begin_turn`, engine-internal, no LLM call): roll
   deterministic 2d6 (existing seeded scheme), move, GO salary, tax / rent /
   own-square resolution, bankruptcy handling. Every event appends a line to
   `state.events` (full list, never truncated) and is broadcast to all seats.
2. **Buy decision** — only if landed on an unowned buyable square AND cash ≥
   price (if unaffordable: auto-event "cannot afford", skip to 3).
3. **Trade step** (same player, once per turn): player may propose **one**
   trade to one counterparty, or pass.
4. If proposed: counterparty responds accept/reject (with optional message).
   Accepted trades execute atomically. Then next player's `_begin_turn`.

Bankruptcy at any point removes the player; last solvent player wins
immediately. At `turn_cap`: richest by net worth (cash + list prices) wins;
ties broken by higher cash, then lower seat index (documented).

### Decision grammars (strict; ParseError on violation → retry/void policy)

- **Buy phase** — exactly one `<decision>` tag required:
  `<decision>buy</decision>` | `<decision>decline</decision>`
  ("I do not want to buy" with no tag = ParseError, never a purchase.)
- **Trade-propose phase** — either `<no_trade/>` (or `<no_trade></no_trade>`)
  or one `<trade>` block:
  ```
  <trade>
    <to>P2</to>
    <give_props>orange1,rr1</give_props>   (optional, comma-separated ids)
    <give_cash>50</give_cash>              (optional, int >= 0)
    <want_props>red2</want_props>          (optional)
    <want_cash>0</want_cash>               (optional)
    <message>free text — promises live here</message>  (optional)
  </trade>
  ```
  Validity (violations are ParseErrors with explanatory text): recipient
  exists, is not self, not bankrupt; at least one of give/want non-empty;
  all `give_props` owned by proposer; all `want_props` owned by recipient;
  `give_cash` ≤ proposer cash; cash values non-negative integers; property
  ids must exist.
- **Trade-respond phase**:
  `<response>accept</response>` | `<response>reject</response>` plus
  optional `<message>...</message>`.
  If the responder accepts but holds less cash than `want_cash`, the engine
  converts to a rejection with event "trade failed: insufficient funds".

Action dicts keep legacy names: `propose_trade`, `accept_trade`,
`reject_trade`, `buy`, `decline`, plus new `no_trade`.

### Prompt content (`render_prompt`) — user-enumerated, binding

Every decision prompt shows: turn / turn_cap and phase; **own cash**; own
position; own properties; **every player's board position**; **full
ownership table** (each buyable square: prop_id, group, price, rent /
full-group rent, owner or bank); bankrupt list; last 12 event lines; the
phase's exact reply format with a literal example. During trade-respond:
the full offer + proposer's message. **Not shown: other players' cash or
net worths.**

### `legal_actions` (for RandomPlayer/tests only; never rendered to LLMs)

buy phase → `[{decline},{buy if affordable}]`; trade-propose →
`[{"type":"no_trade"}]`; trade-respond → `[accept, reject]`.

### Observations / logging

Each step's new `state.events` lines broadcast to all seats as
`Obs(payload={"type": "event", "text": line})` (also god-logged), so players
see the economic history and the transcript builder can render it. Trade
proposals/responses broadcast to all seats (offers are public table talk),
including the free-text messages.

## Workstream D — Analyzer rework (`scripts/analyze_game_steering.py`)

### Objective metrics
- Detect `aborted` logs → per-match row `{label, seed, aborted: true,
  aborted_player, model, steering}`; excluded from all other metrics;
  `cancel_rate` per condition in aggregate + console.
- `coop_side_win` = **None** (not 0.0) for games without a defined
  cooperative side (all except ONW/SH).
- Monopoly adds: `n_trades_proposed`, `n_trades_completed`,
  `n_trades_rejected`. Message/alliance metrics are None for Monopoly.
- Legacy metrics (messages, alliances, `n_fallback` for old logs) retained
  for ONW/SH/Risk; whisper dedup kept.

### Game-aware full transcripts (`build_transcript(recs, game)`)
- **ONW**: unchanged (roles, messages, votes with targets, alliances,
  outcome).
- **Secret Hitler**: votes rendered as `P0 votes ja/nein`; add
  `P0 nominates P2 as chancellor`, `policy enacted: <team>`, veto lines;
  messages/alliances/outcome as before.
- **Risk**: add attack lines (attacker, target territory, owner, armies)
  and elimination events; messages/alliances/outcome as before.
- **Monopoly**: event lines (rolls, buys, rents, bankruptcies) + full trade
  dialogue with messages + outcome.
- Must run without error on the existing v1/v2 production log dirs (action
  schemas are unchanged; this is required for re-analysis of old runs).

### Judge calls
- Truncation `transcript[:9000]` replaced by: send full transcript; if
  > 60,000 chars, send first 20,000 + `[... N chars omitted ...]` + last
  40,000 (outcome always visible).
- Pass 1 (unchanged definitions): cooperation / trust / aggression indices.
- Pass 2 **promise ledger** (new; same Sonnet judge, LLM-judge-never-regex
  rule): extract every promise/commitment:
  ```json
  {"promises": [{"by": "P0", "to": ["P2"], "turn": 12,
                 "promise": "<what was committed>",
                 "status": "kept|broken|unresolved",
                 "evidence": "<quoted transcript line(s)>"}]}
  ```
  Per-match: `n_promises`, `n_kept`, `n_broken`,
  `renege_rate = n_broken / (n_kept + n_broken)` (None when denominator 0).
  Aggregated mean/CI per condition. Applies to all four games.
- CLI: `--no-judge` skips both passes; new `--no-promises` skips pass 2.
- `scripts/steering_games_stats.py`: skip aborted rows; add the new metric
  keys (trades, promises, renege_rate, cancel_rate) to its comparisons.

## Testing (all new/changed behavior)

- `tests/play/test_runner_parse_policy.py` (new): scripted garbage player →
  exactly 4 attempts, `parse_error` ×3 + `aborted` record with required
  fields, no `terminal` record, no `fallback` record type anywhere;
  recovery case (garbage ×2 then valid) completes normally; corrective
  line appears in the LLM player's next prompt.
- `tests/play/test_monopoly.py` (rewrite): grammar accept/reject cases incl.
  negations ("I do not want to buy", "I will not accept" → ParseError or
  correct rejection, never the opposite action); every validity rule; trade
  execution incl. insufficient-funds auto-reject; auto-roll event emission;
  prompt contents (ownership table present, others' cash absent, formats
  shown); bankruptcy transfer; cap winner + documented tie-break.
- `tests/play/test_transcripts.py` (new): per-game fixture logs → SH ja/nein
  + nominate/enact lines, Risk attack lines, Mono trade dialogue, ONW
  regression; 60k head+tail windowing; legacy whisper dedup.
- `tests/play/test_promise_judge.py` (new): mocked client → ledger parsed,
  metrics computed; malformed judge JSON → None row, no crash.
- Update `tests/test_play_harness.py` (drop Diplomacy) and any other
  imports; full suite green.

## QA gates

1. `python3 -m pytest tests/ -v` green (known pre-existing
   network-dependent failures in test_generator.py excepted).
2. `grep -ri diplomacy game_theory_llm/ scripts/ tests/` → zero live-code
   hits (docs/ and data/ excepted).
3. New analyzer end-to-end (`--no-judge`) on
   `data/runs/game_steering_v2` and `data/runs/sh_steering_v1` without error.
4. E2E smoke: scripted deterministic "LLM" player completes a full Monopoly
   match with ≥1 trade proposed and accepted; scripted garbage player match
   aborts after 4 attempts with correct log records.

## Out of scope

- Re-running any steering experiment (separate cost decision; Modal app
  must be redeployed before any new run — the workers bake in engine code).
- ONW / Secret Hitler / Risk engine mechanics (only runner policy + analyzer
  changes touch them).
- The AAAI-27 paper: frozen on the old data and methodology.
- Counter-offers, multi-item auction mechanics, mortgage/houses — YAGNI.
