# Cooperation-Measurement Audit — What the Judge and Metrics Actually See

- Date: 2026-07-22. Companion to `game_mechanics_audit.md` (2026-07-21): that audit
  covered engine rules/hidden-info/metric plumbing; this one asks the construct-validity
  question — **when we report "cooperation" in these games, what is actually being
  measured?** Focus requested: Monopoly and the cooperation-measurement pipeline.
- Method: direct empirical reproduction. The exact `build_transcript()` /
  `objective_metrics()` code from `scripts/analyze_game_steering.py` (post
  whisper-dedup fix, commit e517500) was run over every production log in
  `data/runs/{mono_steering_v2, risk_steering_v1, dip_steering_v1, game_steering_v2,
  sh_steering_v1}`, and each judge transcript was categorised line-by-line and checked
  against the judge's `transcript[:9000]` truncation window. All counts below are
  full-run counts, not spot checks.

## The measurement stack (three layers)

1. **Judge indices** (cooperation/trust/aggression, 0–100): Claude Sonnet 4.6 reads
   `build_transcript(recs)[:9000]` per match. `build_transcript` renders ONLY:
   one god-view setup line, `say`/`whisper` messages, alliance lifecycle events,
   actions of type `vote`, and the terminal OUTCOME line.
2. **Objective alliance metrics**: `n_proposed/accepted/betrayed`, `formation_rate`,
   `betrayal_rate` — from the terminal `alliance_summary`, which buckets every
   accepted alliance as *betrayed* iff any `betrayed` event was emitted for it,
   else *honored* (`alliances.py:338-345`).
3. **Message/outcome metrics**: `public_msg_ratio`, `msgs_per_match`,
   `coop_side_win` (defined only for ONW=village, SH=liberal; all other games
   silently score 0.0).

## Headline findings

| # | Finding | Games | Severity |
|---|---|---|---|
| 1 | Judge indices measure **table-talk, not play**: no economic/military/order action is ever rendered into the judge transcript in any game | all 5 | Critical (construct validity) |
| 2 | Judge's head-truncation `transcript[:9000]` silently drops the late game **and the outcome** for long matches | SH 124/125, Mono 77/100, ONW 20/125 | Critical |
| 3 | SH votes render as `"P0 VOTES PNone"` — schema mismatch (`target` vs `ja`), so the judge sees that ballots happened but **never a single vote direction** | SH | Critical (paper) |
| 4 | Monopoly trades: **145/145 `propose_trade` actions across the full v2 run originate from parse-failure fallback**, never from a model decision | Mono | Critical (already flagged; now full-run-quantified) |
| 5 | Monopoly `betrayal_rate ≡ 0` and `honour ≡ 100%` **by construction** (`judge_alliance` is a no-op, and unbetrayed ⇒ honored) | Mono (Dip moot: 0 accepted) | Important |
| 6 | Risk "aggression_index" judged without the judge seeing a single attack (76 attacks in 100 matches, 0 rendered) | Risk | Important |
| 7 | Models are told trades are "listed in your actions" but `LLMPlayer` renders only `render_prompt` — the candidate trades are **never shown and cannot be selected** (no grammar) | Mono | Important |

## Per-game evidence (full production runs)

### Monopoly (`mono_steering_v2`, 100 matches)

- **Judge transcript line mix**: 20,079 public msgs, 902 whispers, 1,451 alliance
  lines, 100 god + 100 outcome lines, 33 other. **Zero** of the 4,092 rolls,
  774 buys, 751 declines, 145 trade proposals, 71 accepts, 74 rejects are rendered.
  The cooperation/trust/aggression indices for Monopoly are ratings of
  *conversation about* cooperation.
- **Truncation**: median transcript 15,651 chars (max 127,535) vs the 9,000-char
  judge window → the OUTCOME line was inside the window in only **23/100** matches.
  In 77% of matches the judge rated the opening talk and never saw who won —
  and the sampled transcripts show that opening talk is highly degenerate
  (repeated identical "balanced approach / monopolies" pleasantries, literal
  `public message` prefix echoes, one seat repeating the same line 5×).
- **Trades**: 145 `propose_trade` total; **all 145 immediately preceded by a
  `fallback` record** (parse-failure random draw). 126 of the 145 are in
  `coop_a-4` — the condition with the 58.8% parse-error rate — i.e. trade
  activity is a *symptom of format collapse*, not cooperation. The 71
  accepts / 74 rejects are "parsed", but via naive substring matching
  (`"accept" in text`, negation-blind) in response to random proposals.
- **Alliances**: 870 proposed, 201 accepted, 140 broken, **0 betrayed,
  201 honored — honored == accepted exactly**. With `judge_alliance` returning
  `[]`, no betrayed event can ever fire, so every accepted pact lands in
  `n_honored` (`alliances.py:340-341`). Any betrayal/honour contrast for
  Monopoly is a structural constant, not a behavioral measurement.
- `coop_side_win` is a dead column (game not in `COOP_SIDE`; always 0.0).
- Prompt/action mismatch: `_negotiation_prompt` says "Available trade proposals
  are listed in your actions", but the LLM prompt contains only
  `render_prompt` output; `nego_parse` has no trade tag and `parse_action`
  in the roll phase hard-codes `{"type":"roll"}` — a real model **cannot**
  express a trade even if it wants to.

### Secret Hitler (`sh_steering_v1`, 125 matches — feeds the current paper numbers)

- **Votes are informationless to the judge**: 6,709 vote lines all render as
  `P<i> VOTES PNone` because `build_transcript` reads `action["target"]`
  (ONW's schema) while SH votes are `{"type":"vote","ja":bool}`.
- **Truncation is near-total**: median transcript 22,707 chars (max 64,614);
  the OUTCOME line fell inside the 9,000-char judge window in **1/125** matches.
- Also invisible to the judge: all 1,361 nominations, 752 discards, 751 policy
  enactments. Net: the paper's SH cooperation/trust/aggression indices are
  ratings of the *first ~40% of table-talk*, with no vote directions, no
  governments, no policies, and no outcome. (`coop_side_win` / liberal win
  rate is unaffected — it reads the terminal record directly.)
- Any SH v2 re-analysis run through the same analyzer inherits both problems
  unchanged — the v2 engine fixes (secret ballot) do not touch the judge path.

### One Night Werewolf (`game_steering_v2`, 125 matches — clean re-run, feeds the paper)

- The healthiest pipeline: votes render correctly with targets (625 lines),
  god-view roles line gives the judge the hidden-role context, and the OUTCOME
  was inside the window in 105/125 matches (median 7,411 chars). Residual:
  20/125 matches judged without their endgame/outcome.
- Night actions (130 wolf_acknowledge, 70 tm_swap, …) are not rendered — 
  acceptable, since the setup god-view line already gives the judge the roles.

### Risk (`risk_steering_v1`, 100 matches)

- All transcripts fit the window (median 2,587 chars, 100/100 outcomes visible).
- But the judge transcript contains **no military actions at all**: 76 attacks,
  300 deploys, 198 fortifies, 19,802 end_turns — none rendered. The
  aggression_index for Risk was judged from 1,017 public messages + 143
  whispers + 379 alliance lines. (Substantively, the table also barely fought:
  76 attacks across 100 matches.)
- Alliances: 376 proposed, **1** accepted — formation/betrayal columns are
  effectively no-data, consistent with the steering report's "near-zero
  alliances" disclosure.

### Diplomacy (`dip_steering_v1`, 100 matches)

- 1,848 `submit_orders` (the entire strategic content, itself ~95%
  fallback-authored per the mechanics audit) — none rendered to the judge;
  transcript = talk + 422 alliance-propose lines + outcome. 97/100 outcomes
  inside the window.
- 0 alliances ever accepted → formation_rate 0, betrayal_rate undefined
  everywhere.

## Implications for published numbers

- **Paper (aaai27.tex / steering report SH rows)**: the SH judge indices are
  early-game-talk measures (Findings 2+3). The ONW judge indices are
  substantially sound (correct votes, roles context, 84% outcome visibility).
  Win rates in both games are objective and unaffected. If SH v2 numbers are
  swapped into the paper via the current analyzer, the same limitation carries
  over verbatim.
- **Steering report (Monopoly rows)**: cooperation/trust/aggression indices for
  Monopoly are talk-only measures whose strongest contrast (α=−4) sits on top
  of the already-disclosed parse-collapse confound; "trade willingness" as a
  behavioral axis is confirmed dead at full-run scale (145/145 fallback);
  betrayal/honour columns are structural constants and should not be cited as
  behavior.
- Message metrics (`public_msg_ratio`, `msgs_per_match`) and `coop_side_win`
  (ONW/SH) remain valid as defined (post whisper-dedup fix).

## Recommended fixes (NOT applied — decision needed, some invalidate comparability)

1. **Schema-aware transcript**: render SH votes as ja/nein and add per-game
   act lines (SH nominate/enact; Mono buy/trade-accept; Risk attack; Dip order
   digests). Cheap, analyzer-only — but changes the judge's input, so
   judge indices are not comparable across the fix.
2. **Head+tail windowing** instead of `transcript[:9000]` (always keep the
   god line and the final segment incl. OUTCOME), or raise the cap for the
   judge call. Same comparability caveat.
3. **Monopoly trade affordance**: list candidate trades in the prompt and add a
   trade tag to the negotiation grammar (engine change; only meaningful with a
   re-run).
4. **Monopoly honour semantics**: either implement a real `judge_alliance`
   (e.g. nonaggression = no rent-triggering trades against ally? nothing
   mechanically natural exists) or mark honour/betrayal columns N/A for
   Monopoly rather than reporting structural constants.
5. Report `coop_side_win` as N/A (not 0.0) for games without a defined
   cooperative side.
