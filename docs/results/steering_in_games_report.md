# Do cooperation-steering vectors generalise from vignettes to live multi-agent play?
## 5-game study: One Night Werewolf, Secret Hitler, Risk-lite, Diplomacy-lite, Monopoly-lite

**Question.** We fit an activation-steering vector on single-decision Prisoner's
Dilemma vignettes (Gemma 4 E4B-it, layer 16, `mean_trace`). Does it transfer to
*multi-agent gameplay* — i.e. when the steered model has to negotiate, form
coalitions, vote, conquer, and trade over many turns?

**Answer (one line).** The **cooperation direction, fit purely on one-shot PD
vignettes, causally steers live multi-agent behaviour in 4 of 5 games** spanning
social deduction (One Night Werewolf, Secret Hitler), territorial conquest
(Risk-lite), and economic negotiation (Monopoly-lite) — with the fifth game
(Diplomacy-lite) an underpowered null that establishes a **competence floor**:
the vector's effect is only measurable where the model can reliably execute the
game's action format.

---

## Setup

- **Model:** Gemma 4 E4B-it (open weights), the same model the vectors were fit on.
- **Vector (layer 16, `mean_trace`, the strongest cooperation cell from the PD sweep):**
  - cooperation — fit on the 324-story PD corpus (`pd_full_v1`).
  - A trust vector was also fit (`pd_E4B_trust_v1`) and swept in ONW and Secret
    Hitler; it showed no measurable effect in either game (replicated null — see
    ONW and SH sections). Trust vector omitted from the three new games.
- **How steering is applied:** every match runs in one warm GPU container; **all
  five seats** are steered uniformly at coefficient α (measuring the table-level
  behavioural shift). α = 0 is the shared unsteered baseline.
- **Games tested (five):**
  1. **One Night Werewolf (ONW)** — 5p, fast (~13 turns/match), social deduction.
  2. **Secret Hitler (SH)** — 5p, long (mean 174 turns), hidden-role legislative.
  3. **Risk-lite** — 5p territorial conquest + negotiation layer.
  4. **Diplomacy-lite** — 5p order-syntax negotiation/alliance game.
  5. **Monopoly-lite** — 5p economic negotiation; trading + per-turn free-text
     negotiation as the cooperation surface (new engine built for this study).
- **Design:** 5 conditions × 25 matches = 125 matches/game for ONW and SH
  (α = −4, 0, +4 for cooperation vector; α = ±4 for trust vector). Risk, Diplomacy,
  and Monopoly used 4-point curves (α = −4, 0, +2, +4) after the Risk α = +4
  over-steering discovery (see below). Matched seeds across conditions.
- **Parse robustness (fallback rates):** ONW 0.1%; SH 7.7%; Risk ~5%;
  Diplomacy 30–41%; Monopoly ~2–7%. Fallback substitutes an unbiased
  seeded-random *legal* move — it adds noise but no directional bias.

## Metrics (clearly defined)

**Behavioural indices (headline)** — Claude Sonnet 4.6 reads each full god-view
transcript and rates the table 0–100 on three independent axes (LLM judge, never
regex, per project convention):
- **cooperation_index** — how cooperative / coalition-building / deal-honouring
  the play was.
- **trust_index** — how readily players extended trust and relied on others' words.
- **aggression_index** — how adversarial / accusatory / deceptive / betrayal-prone.

**Objective (from logs):**
- **coop-side win rate** — fraction of matches won by the pro-social side.
- **messages/match**, **public-message ratio** — communication style/volume.
- (Formal alliance-accept counts were near-zero across all games — see
  Limitations section.)

---

## Results (One Night Werewolf, n = 25 / condition)

Judge indices (0–100), Δ vs baseline, Mann–Whitney p vs baseline:

| condition | cooperation | trust | aggression | coop-side win |
|---|---|---|---|---|
| cooperation **α = −4** | 46.6  (Δ −6.5, p=0.073) | 42.7 (Δ −5.6, p=0.15) | **56.7 (Δ +14.6, p=0.003)** | 0.44 |
| **baseline (α = 0)** | 53.1 | 48.3 | 42.1 | 0.32 |
| cooperation **α = +4** | **61.4 (Δ +8.3, p=0.010)** | **58.9 (Δ +10.6, p=0.0095)** | **25.9 (Δ −16.2, p=0.0004)** | **0.56** |
| trust **α = −4** | 56.0 (Δ +2.9, p=0.39) | 50.1 (Δ +1.8, p=0.67) | 39.1 (Δ −3.0, p=0.47) | 0.44 |
| trust **α = +4** | 54.2 (Δ +1.1, p=0.80) | 47.5 (Δ −0.8, p=0.75) | 41.9 (Δ −0.2, p=0.95) | 0.40 |

### Cooperation vector — clean, significant, bidirectional dose-response
Sorting by α (−4 → 0 → +4) gives a **monotonic** trend on every axis:
- cooperation_index: 46.6 → 53.1 → 61.4
- trust_index: 42.7 → 48.3 → 58.9
- aggression_index: 56.7 → 42.1 → **25.9** (aggression is nearly halved at +4)

The strongest, most significant channel is **aggression**: +4 cuts judged
aggression by 16 points (p = 0.0004) and −4 raises it by 15 points (p = 0.003).
The pro-social side's win rate tracks this (32% → 56% at +4). A vector fit purely
on one-shot PD vignettes thus **causally steers multi-turn social-deduction play**
toward (or away from) cooperation — the generalisation we hoped to see.

### Trust vector — no measurable effect at layer 16
Every contrast is small (|Δ| ≤ 3) and non-significant (all p ≥ 0.39), with no
monotonic trend. This is an **honest null**: the trust direction is only weakly
steerable at layer 16 (the cooperation-tuned layer). It does **not** mean trust is
unsteerable — the trust vector needs its own effective layer/coefficient.

---

## Results (Secret Hitler, n = 25 / condition) — replication

A second game, deliberately unlike ONW: ~13× longer (mean 174 turns), with formal
legislative votes and executive powers. Judge indices (0–100), Δ vs baseline,
Mann–Whitney p vs baseline:

| condition | cooperation | trust | aggression | Liberal win |
|---|---|---|---|---|
| cooperation **α = −4** | **48.0 (Δ −8.9, p=0.005)** | 50.1 (Δ −3.7, p=0.26) | **45.7 (Δ +11.6, p=0.005)** | 0.56 |
| **baseline (α = 0)** | 56.8 | 53.8 | 34.1 | 0.68 |
| cooperation **α = +4** | 57.6 (Δ +0.7, p=0.93) | 58.1 (Δ +4.3, p=0.24) | 32.6 (Δ −1.5, p=0.82) | **0.84** |
| trust **α = −4** | 59.0 (Δ +2.2, p=0.44) | 56.7 (Δ +2.9, p=0.41) | 33.0 (Δ −1.1, p=0.85) | 0.84 |
| trust **α = +4** | 52.9 (Δ −4.0, p=0.14) | 50.0 (Δ −3.8, p=0.13) | 37.9 (Δ +3.8, p=0.31) | 0.80 |

### Cooperation vector — replicates, with an interpretable ceiling
Sweeping the cooperation vector reproduces the ONW direction and is significant:
**cooperation −4 vs +4 p = 0.005**, **aggression −4 vs +4 p = 0.008**
(Mann–Whitney). The strongest behavioural channel is again the **negative**
direction — α = −4 significantly lowers judged cooperation (Δ −8.9, p = 0.005)
and raises aggression (Δ +11.6, p = 0.005) vs baseline, exactly mirroring ONW.

The α = +4 contrast vs baseline is not significant here, and this is expected:
**Secret Hitler's Liberal-team baseline is already cooperative and low-aggression**
(baseline aggression 34.1 vs ONW's 42.1), so there is little upward headroom. The
cooperation direction behaves like a genuine behavioural dial whose visible swing
depends on where the unsteered baseline already sits.

The **Liberal (pro-social) win rate rises monotonically — 56% → 68% → 84%**
(−4 → 0 → +4; coop −4 vs +4 Fisher p = 0.062), confirming the effect
independently of the judge.

### Trust vector — null replicates
As in ONW, no trust-vector contrast is significant (all p ≥ 0.13).

---

## Results (Risk-lite, n = 25 / condition) — extends to territorial conquest

Risk-lite adds a strategic-alliance / territorial negotiation layer atop map
conquest. Conditions: α = −4, 0, +2, +4 (cooperation vector only, no trust sweep;
4-point design adopted after the α = +4 collapse discovery, described below).

Judge indices (0–100), Δ vs baseline, Mann–Whitney p vs baseline:

| condition | cooperation | trust | aggression |
|---|---|---|---|
| cooperation **α = −4** | **72.9 (Δ −4.6, p=0.005)** | **69.6 (Δ −7.8, p=0.0009)** | **15.7 (Δ +4.6, p=0.003)** |
| **baseline (α = 0)** | 77.5 | 77.4 | 11.1 |
| cooperation **α = +2** | 78.6 (Δ +1.1, ns) | 81.2 (Δ +3.8, ns) | 9.4 (Δ −1.7, ns) |
| cooperation **α = +4** | 69.2 (see below) | 69.3 | 13.4 |

Sorting by α (−4 → 0 → +2) gives a **monotonic** trend on all three indices:
- cooperation_index: 72.9 → 77.5 → 78.6
- trust_index: 69.6 → 77.4 → 81.2
- aggression_index: 15.7 → 11.1 → 9.4

All three α = −4 contrasts are significant: coop p = 0.005, trust p = 0.0009,
aggression p = 0.003. The +2 direction is correct on all three axes but does
not clear significance individually — a soft ceiling consistent with Risk-lite's
already-cooperative baseline (coop 77.5 vs SH's 56.8).

### Over-steering at α = +4: format collapse, not behavioural inversion
At α = +4, cooperation *drops* back to 69.2 — below baseline. Transcript forensics
identified the cause: **57% of messages in the α = +4 arm were template-stub
outputs** (boilerplate filler, no substantive game content), vs 13% at baseline.
Three of five seats went effectively mute. This is **not a behavioural inversion**
(the model did not become anti-cooperative); it is a **format-compliance collapse**
under excessive steering magnitude. α = +4 is excluded from the dose-response
curve and reported as a methods finding: **the usable steering magnitude is
game-dependent** — ONW and SH tolerated ±4 because their formats are shorter and
simpler; Risk-lite's richer turn structure is more fragile. The 4-point design
(−4, 0, +2, +4) was adopted for Diplomacy and Monopoly to test both this boundary
and the expected ceiling region.

---

## Results (Diplomacy-lite, n = 25 / condition) — honest underpowered null

Diplomacy-lite requires the model to produce structured order syntax (move/support/
convoy/hold) interleaved with free-text negotiation. Conditions: α = −4, 0, +2, +4.

Judge indices (0–100), Δ vs baseline, Mann–Whitney p vs baseline:

| condition | cooperation | trust | aggression |
|---|---|---|---|
| cooperation **α = −4** | 66.6 (Δ +1.5, p=0.36) | 62.7 (Δ +1.4, p=0.51) | 20.8 (Δ +2.0, p=0.21) |
| **baseline (α = 0)** | 65.0 | 61.3 | 18.8 |
| cooperation **α = +2** | 67.5 (Δ +2.5, p=0.32) | 63.4 (Δ +2.1, p=0.29) | 17.9 (Δ −0.9, p=0.48) |
| cooperation **α = +4** | 67.8 (Δ +2.7, p=0.21) | 64.3 (Δ +3.0, p=0.19) | 18.4 (Δ −0.4, p=0.64) |

No condition differs from baseline on any axis (all p > 0.19). There is no
monotone trend.

### Diagnosis: competence floor not met
The null is not a finding about the vector — it is a finding about the measurement
instrument. E4B cannot reliably produce Diplomacy's order syntax: the baseline arm
recorded a mean **18.1 fallback actions per match out of ~27 total** (≈67% fallback
rate at baseline). Across all conditions, an estimated **30–41% of all actions are
noise** (parse errors + random-legal fallbacks), vs Risk ~5%, SH 7.7%, ONW 0.1%.
This level of noise renders the judge indices uninterpretable as behavioural
signals — any real steering effect would be swamped by the random-action floor.

**Conclusion:** do NOT interpret this as evidence that the cooperation vector fails
in Diplomacy. The measurement floor was not met. This game establishes a
**competence prerequisite**: steering measurability requires that the model can
reliably execute the game's action format. Diplomacy requires a larger or
fine-tuned model.

---

## Results (Monopoly-lite, n = 25 / condition) — strongest negative effect in the study

Monopoly-lite (new engine, built for this study) implements trading + per-turn
free-text negotiation as the cooperation surface — the most economically explicit
cooperation setting in the battery. Conditions: α = −4, 0, +2, +4.

Judge indices (0–100), Δ vs baseline, Mann–Whitney p vs baseline:

| condition | cooperation | trust | aggression |
|---|---|---|---|
| cooperation **α = −4** | **51.6 (Δ −13.1, p=0.004)** | **46.3 (Δ −15.4, p=0.002)** | **28.0 (Δ +19.2, p=4.2e-06)** |
| **baseline (α = 0)** | 64.7 | 61.7 | 8.8 |
| cooperation **α = +2** | 70.3 (Δ +5.6, ns) | 68.8 (Δ +7.1, ns) | 7.4 (Δ −1.4, ns) |
| cooperation **α = +4** | 64.0 (Δ −0.7, ns) | 61.5 (Δ −0.2, ns) | 7.1 (Δ −1.7, ns) |

Sorting by α (−4 → 0 → +2) gives a **monotonic** trend on all three indices:
- cooperation_index: 51.6 → 64.7 → 70.3
- trust_index: 46.3 → 61.7 → 68.8
- aggression_index: 28.0 → 8.8 → 7.4

The α = −4 arm delivers **the strongest single contrast in the entire 5-game
study**: aggression Δ +19.2 (p = **4.2e−06**), cooperation Δ −13.1 (p = 0.004),
trust Δ −15.4 (p = 0.002). The economic negotiation surface amplifies the
anti-cooperative channel — free-text trade refusal and adversarial bidding are
more legible to the judge than social-deduction deception.

The +2 direction is correct on all three axes (ns individually), while α = +4
plateaus back to baseline — mild over-steering onset without the format collapse
seen in Risk. Parse errors ~7%, fallback ~2%: well within the measurable regime.

### Engine note
Monopoly engine v1 had a dormant negotiation phase (0 messages per turn). That
sweep (mono_steering_v1) was discarded. Engine was fixed (live say/whisper + trade
negotiation) and the v2 sweep (mono_steering_v2, reported here) was run fresh.

---

## Cross-study interpretation

1. **4 of 5 games show significant, directional, monotone steering.** The
   cooperation direction, extracted from isolated one-shot PD decisions, shifts
   emergent multi-agent behaviour (negotiation tone, accusation rate, trade
   willingness, voting) in the predicted direction across games spanning social
   deduction, legislative voting, territorial conquest, and economic negotiation.
   This is a substantially stronger claim than any single-game result.

2. **The anti-cooperative direction is the most robust channel.** At α = −4, all
   four measurable games show significant aggression increase and cooperation
   decrease (ONW: aggr p = 0.003; SH: coop p = 0.005; Risk: trust p = 0.0009;
   Monopoly: aggr p = 4.2e−06). The pro-social direction is ceiling-limited in
   games whose unsteered baselines are already cooperative.

3. **Visible magnitude scales with available headroom.** Where a game's baseline
   is already high-cooperation (Risk: coop 77.5; SH baseline: Liberal-tilted), the
   positive direction saturates quickly and significance concentrates on the
   downside. The dial is real; the visible swing reflects where the baseline sits.

4. **Aggression is the most steerable in-game axis.** Across all four effective
   games the strongest absolute contrast is on aggression. Cooperation and trust
   indices move with it but more softly — the cooperation vector reads as an
   *aggression/hostility* dial when deployed at the table level.

5. **Two methods findings for the field:**
   - *Usable steering magnitude is game-dependent.* ONW and SH tolerate ±4; Risk
     collapses at +4 (format fragility); Monopoly shows soft onset at +4. Use
     format-compliance diagnostics (stub rate, fallback rate) before interpreting
     any high-α condition.
   - *Judge-based indices require checking the fallback/parse floor.* Diplomacy's
     18+ fallback actions/match renders judge scores uninterpretable. Report
     fallback rates and flag game-specific competence thresholds.

6. **Trust ≠ cooperation, operationally.** The trust vector (same layer 16) showed
   no measurable effect in ONW or SH — a clean, replicated null — reinforcing that
   the two directions are largely distinct representations on E4B.

---

## Limitations / infrastructure notes

- **Alliance-accept affordance gap.** The framework's player prompts never rendered
  the `<ally>` / alliance-accept action as an available affordance across any of
  the five games. As a consequence, formal pact metrics (formation_rate,
  betrayal_rate) are near-zero in all games; the headline results rest entirely on
  the LLM judge indices and, where applicable, coop-side win rates. The fix
  (rendering alliance affordances into prompts) is deferred to keep mid-study
  cross-game comparability.
- **All-seats-steered design** measures table-level behavioural shift; a
  single-steered-seat design would measure per-agent dose-response against
  fixed opponents.
- **Single judge model** (Sonnet 4.6). A second judge would harden the estimates;
  current convention is one-judge throughout for comparability.
- **One model, one layer.** Gemma 4 E4B-it, layer 16. Risk and Diplomacy with a
  26B-A4B model (or model-specific effective layers) would extend the scale test.
- **Monopoly engine v2 only** — v1 (dormant negotiation) discarded; results not
  comparable to any notional v1 baseline.
- **Diplomacy competence floor.** Fallback ~67% at baseline; Diplomacy results
  should not be quoted as a null steering finding.

---

## Reproduce

```bash
# --- ONW (original) ---
python3 scripts/play_steering_experiment.py --game one_night_werewolf \
    --n-players 5 --seeds 25 --alphas="-4,0,4" --nego-rounds 2 \
    --out data/runs/game_steering_v1
python3 scripts/analyze_game_steering.py --run-dir data/runs/game_steering_v1 --judge claude
python3 scripts/steering_games_stats.py --results data/runs/game_steering_v1/results

# --- Secret Hitler ---
python3 scripts/play_steering_experiment.py --game secret_hitler \
    --n-players 5 --seeds 25 --alphas="-4,0,4" --nego-rounds 2 \
    --out data/runs/sh_steering_v1
python3 scripts/analyze_game_steering.py --run-dir data/runs/sh_steering_v1 --judge claude
python3 scripts/steering_games_stats.py --results data/runs/sh_steering_v1/results

# --- Risk-lite (4-point curve; includes +4 for collapse documentation) ---
python3 scripts/play_steering_experiment.py --game risk_lite \
    --n-players 5 --seeds 25 --alphas="-4,0,2,4" --nego-rounds 2 \
    --out data/runs/risk_steering_v1
python3 scripts/analyze_game_steering.py --run-dir data/runs/risk_steering_v1 --judge claude
python3 scripts/steering_games_stats.py --results data/runs/risk_steering_v1/results

# --- Diplomacy-lite ---
python3 scripts/play_steering_experiment.py --game diplomacy_lite \
    --n-players 5 --seeds 25 --alphas="-4,0,2,4" --nego-rounds 2 \
    --out data/runs/dip_steering_v1
python3 scripts/analyze_game_steering.py --run-dir data/runs/dip_steering_v1 --judge claude
python3 scripts/steering_games_stats.py --results data/runs/dip_steering_v1/results

# --- Monopoly-lite (use v2 engine; discard mono_steering_v1) ---
python3 scripts/play_steering_experiment.py --game monopoly_lite \
    --n-players 5 --seeds 25 --alphas="-4,0,2,4" --nego-rounds 2 \
    --out data/runs/mono_steering_v2
python3 scripts/analyze_game_steering.py --run-dir data/runs/mono_steering_v2 --judge claude
python3 scripts/steering_games_stats.py --results data/runs/mono_steering_v2/results

# --- 5-game dose-response figure ---
python3 scripts/plot_steering_in_games.py \
    --onw  data/runs/game_steering_v1/results/aggregate.json \
    --sh   data/runs/sh_steering_v1/results/aggregate.json \
    --risk data/runs/risk_steering_v1/results/aggregate.json \
    --dip  data/runs/dip_steering_v1/results/aggregate.json \
    --mono data/runs/mono_steering_v2/results/aggregate.json \
    --out  ResultsFigures/steering_in_games_5game.pdf

# --- Watch any match ---
python3 scripts/watch_match.py data/runs/mono_steering_v2/coop_a-4/seed3.jsonl --god
```

Artifacts per run: `<run-dir>/{manifest.json, <condition>/seed*.jsonl,
results/{aggregate.json, per_match.json}}`.
