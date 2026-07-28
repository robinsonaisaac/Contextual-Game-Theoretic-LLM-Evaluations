# Pre-registration: does the cooperation vector remove cooperation one-sidedly?

**Written 2026-07-28, BEFORE the confirmatory data exists.** Committed prior to launch;
the commit hash of this file predates the run directory `data/runs/pg_confirm_v1`.

## Why this exists

An exploratory 90-match battery (`data/runs/pg_battery_v1`) found that steering
Gemma 4 E4B with the PD-fit cooperation vector raised free-riding on the
anti-cooperative arm (0.250 → 0.415) while the pro-cooperative arm sat at
baseline (0.205). That result is **not trustworthy as it stands**, for two
reasons that are entirely about how it was produced:

1. **Post-hoc DV selection.** Free-ride rate was chosen after inspecting the
   secondary measures. The pre-specified DV for that run — mean contribution
   rate — was a null in which the two opposite-signed arms were *identical*
   (0.373 vs 0.369, p = 1.000).
2. **Optional stopping.** Tests were run repeatedly as the battery filled.
   Continuous peeking inflates Type I error; the reported p-values are not the
   p-values they appear to be.

This run fixes every degree of freedom in advance.

## Design

- Game: `public_goods`, 5 seats, 8 rounds, endowment 20, multiplier 2.0,
  messaging 2 negotiation rounds x 2 messages (identical to the exploratory run).
- Model: `google/gemma-4-E4B-it`; vector `pd_full_v1`, layer 16, `mean_trace`.
- Cells: `baseline` (alpha=0), `k5_a-4` (all 5 seats, alpha=-4),
  `k5_a+4` (all 5 seats, alpha=+4).
- **n = 30 completed matches per cell.**
- **Seeds 1000-1029** — disjoint from the exploratory run's 0-29, so this is an
  out-of-sample test rather than a re-analysis.

## Primary hypothesis

**H1.** Free-ride rate (share of individual contribution choices strictly below
20% of endowment, computed per match) differs between `k5_a-4` and `k5_a+4`.

- Test: **Mann-Whitney U, two-sided, alpha = 0.05.**
- This contrast is chosen as primary because it does not involve the baseline
  cell, so it cannot be confounded by anything specific to the unsteered
  condition.
- Empirical power at n=30/arm, bootstrapped from the exploratory
  distributions: **0.979** (rank-biserial r = 0.59). Note this is an
  *optimistic* estimate: a post-hoc-selected effect is typically inflated, so
  true power is lower.

## Secondary hypotheses (Holm-corrected across the three)

- **H2.** `k5_a-4` free-ride rate > `baseline`. *Predicted: significant.*
- **H3.** `k5_a+4` free-ride rate vs `baseline`. **Predicted: NOT significant.**
  This is the one-sidedness claim — the pro-cooperative direction does not
  install cooperation.
- **H4.** Mean contribution rate, `k5_a-4` vs `k5_a+4`. **Predicted: NOT
  significant** (replicating the exploratory p = 1.000): the effect lives in
  the low tail, not the average.

## What would falsify the claim

- H1 non-significant -> the free-riding effect does not replicate out of sample;
  the exploratory result was a post-hoc artefact.
- H3 **significant in the cooperative direction** -> the effect is two-sided and
  the "removes but does not install" framing is wrong.
- H4 significant -> the effect is not confined to the low tail after all, and
  the exploratory null on mean contribution was itself the artefact.

## Analysis and stopping rules

- The analysis is `scripts/confirm_pg_freeriding.py`, committed before launch.
  It is run **once**, after all 90 matches have been collected. No variant of
  it will be run on partial data.
- **No interim looks at the dependent variable.** The progress monitor for this
  run reports match counts and error counts only, and is forbidden from
  printing any DV value.
- Matches that abort (no terminal record) are excluded; if fewer than 30 per
  cell complete, additional seeds are drawn *in ascending order from 1030* to
  top up to exactly 30, and this is reported. Cells are never trimmed to equal
  size by discarding completed matches.
- No DV, test, or cell will be added after launch. Anything not listed above is
  exploratory and will be labelled as such.
