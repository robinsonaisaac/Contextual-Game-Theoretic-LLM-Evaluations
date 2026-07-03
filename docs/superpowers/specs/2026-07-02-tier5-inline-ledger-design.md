# Tier-5: Domain-General Inline Bookkeeping ("ledger") — Design

**Date:** 2026-07-02
**Branch:** `feature/activation-steering` (continues the gt_rlvr line of work)
**Status:** design approved; pending spec review → writing-plans

## 1. Goal & hypothesis

The Tier 1–4 post-mortem localized the long-horizon reasoning failure precisely: local
computation is near-perfect (0.996 env-managed), but the model cannot run a **self-directed
bookkeeping loop** — it loses control state (whose turn / what's done / what's next) over
60+ sequential steps. Outcome-RL could not teach this because the behavior was never in the
reachable set (d6 GRPO groups all-fail → no gradient); prompting could not install it
(memory 0.006, guided 0.025, forced-chunk 0.000 vs plain 0.163).

**Hypothesis:** the bookkeeping loop is a *trainable, domain-general technique*. SFT on
algorithmically-generated **inline state-ledger traces** across diverse task families puts
the behavior in the reachable set; RL under the model's own rollouts hardens its
reliability; and because the *protocol* (not the task) is the invariant, the technique
transfers to unseen long-horizon domains.

**Honest alternative this experiment can also confirm:** if SFT fails to install the
behavior even in-domain at depth, the "in-context control is a capability property"
conclusion strengthens — a publishable negative.

## 2. Decisions on record (this brainstorm)

- **Form: inline ledger** — the state lives inside one long generation (portable to any
  downstream task, no harness at inference). Not the multi-round harness (infrastructure,
  not a technique). Rationale: d6 failures at 8k tokens were discipline, not budget (a full
  d6 ledger ≈ 1.6k tokens).
- **Families: 4 train / 2 held-out** + ≥3 real benchmarks (standing ≥3-benchmark rule).
- **Recipe: SFT → RL, both phases planned.** SFT = reachable-set fix; RL = reliability/
  recovery hardening (SFT-on-perfect-traces has exposure bias, unforgiving at
  acc ≈ p_local^N). A cheap **eval checkpoint** between phases handles two edge cases:
  SFT saturates (skip RL) or SFT installs nothing in-domain (stop; capability-property
  conclusion).
- **Recovery traces in SFT** (~10–15%): injected ledger corruption + explicit
  checkpoint-verify-and-correct, so recovery is taught, not hoped for.
- **Protocol = semantic invariants, surface via LLM naturalization** (2026-07-02 revision):
  a fixed grammar risks token-binding (cf. the SAE "cooperate"-word-detector artifact).
  Traces are generated as skeletons, ~75% rephrased naturally by Sonnet under rotated style
  seeds with **rejection-sampled round-trip verification**; ~25% stay canonical for the
  Phase-2 dense reward. Chosen over a hand-built renderer bank (better diversity, less code).
- **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (comparability with Tiers 1–3), LoRA on
  Tinker (billing confirmed live 2026-07-02; auth + training-client probe OK).

## 3. The ledger protocol — semantic invariants, not a fixed grammar

**Lesson applied (from our own SAE result):** train on one surface form and the model
learns the surface form (cf. the "cooperate"-word-detector artifact). The skill is
therefore specified as **five semantic invariants** that every trace must exhibit *in any
syntax*; surface form is deliberately varied.

The invariants (the contract):
1. **Externalize established facts** compactly as you go — never hold them in your head.
2. **Ground each step in recorded facts** — state what is computable now and from which
   recorded facts (the control decision the model loses at depth, externalized as text).
3. **Small bounded steps** — never re-derive long chains.
4. **Periodically re-establish state** — restate the full known-set at intervals (~8–12
   steps; the inline analogue of env-managed state refresh, the regime that scored 0.988).
5. **Answer read off the final state**, not recomputed.

### Trace skeleton (canonical IR)
Generators emit a format-independent **skeleton**: a list of semantic events —
`note(key, value, deps)`, `restate()`, `answer(x)` — plus the gold key-set. All
verification (tests, data QC, and the Phase-2 dense reward) happens at skeleton level:
one verifier, format-independent.

### Surface realization: canonical grammar + LLM naturalization (rephrase-then-verify)
- **Canonical tagged grammar (~25% of traces)** — the original
  `LEDGER / NEXT / NOTE / CHECKPOINT / ANSWER` form, with a deterministic paired parser.
  Kept because Phase-2 RL's dense reward needs cheaply parseable rollouts (see §6).
- **LLM-naturalized traces (~75%)** — Sonnet 4.6 rephrases each skeleton into natural,
  varied prose/format under a style seed rotated per call ("engineer's log", "formal
  derivation", "casual working notes", "markdown state table", "numbered facts F1/F2…"),
  with the hard instruction: vary wording/structure freely, **never alter values, keys, or
  dependencies**.
- **Rejection-sampled faithfulness (non-negotiable):** every naturalized trace goes
  through an LLM **extract-back** pass (pull the claimed `(key, value, deps)` facts and
  restate/answer events out of the text) followed by a **mechanical equality check**
  against the gold skeleton. Traces that do not round-trip exactly are regenerated or
  dropped — corrupted traces would teach corrupted bookkeeping, the exact disease under
  treatment. (Extraction is far easier than generation; the equality check is mechanical,
  so the chain stays trustworthy without regex-judging quality.)
- Estimated naturalization cost: ~3k traces × rephrase+extract ≈ $60–100 (OpenRouter).

`ledger_protocol.py` owns: the skeleton datatypes, the canonical renderer + paired parser,
the skeleton-level verifier, and the naturalize/extract-back/QC pipeline drivers.

## 4. Task families (each: generator → problem + gold answer + gold trace; tunable horizon; seeded)

**Train (4):**
| family | ledger content | horizon knob |
|---|---|---|
| minimax game trees (existing `gametree.py`) | node → value, post-order | depth (d3–d6 ≈ 7–63 nodes) |
| register-machine simulation | register → value per step | program length N |
| graph search (BFS / shortest path) | frontier, visited, dist[] | nodes/edges |
| forward-chaining deduction (Horn rules) | derived-facts set | rule-chain length |

**Held-out (2, never trained, transfer test):** multi-entity object tracking
(entity → location under swaps); constraint scheduling (prereqs → earliest start times).
Generators + verifiers are built for these too (needed for evaluation), but no traces enter
training.

**Dataset composition (~2–4k traces, all round-trip-verified):** horizons spanning
~10–100 ledger steps (cram impossible at the top); **~25% canonical grammar / ~75%
LLM-naturalized** (per §3, families crossed with styles so the behavior is the only
dataset-wide invariant); **~15% short tasks answered directly with no ledger** (teach
*when* to deploy; protect short-form behavior); **~10–15% recovery traces** (corruption →
restate-verify → correct → continue; rendered in both canonical and naturalized forms).

## 5. Phase 1 — SFT

LoRA SFT on Tinker (`.venv-tinker`), loss on completion (trace + answer) only. Reuses the
gametree-SFT precedent scripts. Deliverable: checkpoint + eval-checkpoint report.

**Eval checkpoint (the gate):** in-domain trained families at trained + extrapolated
horizons, temp 0. Outcomes:
- Behavior installed, reliability lags (expected) → proceed to Phase 2.
- Evals saturated → skip RL, go to final eval.
- Ledger not installed even in-domain → STOP; report the negative.

## 6. Phase 2 — RL (GRPO)

- **Where:** trained families at the **cram boundary** — horizons where the SFT model's
  groups are mixed (some succeed), so advantage exists. Curriculum expands horizon as
  reliability rises.
- **Reward:** `answer_correct + λ · ledger_accuracy` where ledger_accuracy = skeleton-level
  per-note correctness (from the verifier). Dense signal exists even when final answers
  fail — the fix for the d6 no-signal problem. (Distinct from the rejected Tier-3 process
  reward, which scored values inside an unstructured cram; this rewards the state actions
  of an explicit protocol.) λ ≈ 0.5 initially, anneal toward outcome-only if
  reward-hacking of NOTEs appears (spray-guard: ledger_accuracy uses precision·recall
  against the gold key-set, not raw match count).
- **Format anchor:** rollout prompts carry a light canonical-format cue so the paired
  deterministic parser extracts the skeleton and the dense reward stays cheap and
  non-hackable. (LLM-extracting free-form rollouts inside the reward loop would be slow,
  noisy, and gameable.) The SFT-side naturalized majority carries the anti-token-binding
  pressure; RL hardens the loop in the canonical skin; transfer of the invariant is what
  §7 measures.
- Infra: `tinker_grpo.py` patterns; single-context env (no multi-round harness).

## 7. Final generalization evaluation (all before/after: base vs SFT vs SFT+RL, temp 0)

1. **In-domain, extrapolated horizons** (e.g., trees d5–d7): success ≥ +15pp where base
   collapses.
2. **Held-out families zero-shot** (object tracking, scheduling): success ≥ +10pp.
3. **Real benchmarks (≥3):** BBH multistep-arithmetic-two, tracking-shuffled-objects,
   dyck-languages (long-horizon state-tracking); **GSM8K as short-form control**
   (degradation ≤ 2pp). Success: uplift on ≥2 of 3.
4. **Attribution:** (a) **unprompted technique-usage rate** in transfer domains — since
   the model chooses its own style, this is an **LLM-judge classification against the five
   §3 invariants** (Sonnet; quality judgment → judge, per project rule), not a grammar
   parse; (b) **suppression ablation**: instruct the trained model not to use any explicit
   state-tracking → gains should shrink toward base. Together these attribute uplift to
   the technique rather than generic fine-tuning.

**Claim discipline:** "generalizable technique" is claimed only if (2) AND (3) pass; (1)
alone is an in-domain result; report whatever comes out.

## 8. New code (small units, each testable)

```
game_theory_llm/reasoning/ledger_protocol.py     skeleton IR + canonical render/parse +
                                                 skeleton-level verifier (+ round-trip tests)
game_theory_llm/reasoning/ledger_tasks/          6 generators (4 train + 2 held-out), each
                                                 emitting skeletons, with
                                                 gold-skeleton-passes-verifier tests
scripts/naturalize_traces.py                     Sonnet rephrase -> extract-back -> equality
                                                 QC (rejection sampling; style seeds)
scripts/build_tier5_ledger.py                    dataset builder (composition per §4)
scripts/tinker_sft_ledger.py                     Phase-1 SFT launcher
scripts/tier5_evalsuite.sh                       eval battery (§5 checkpoint + §7 final)
scripts/tinker_grpo_ledger.py                    Phase-2 GRPO launcher (built at gate time)
```

Reuse: `gametree.py`, BBH/GSM8K eval corpora already in `data/runs/{bbh,capability}`,
Tinker SFT/GRPO script patterns, tier1b/tier2 evalsuite patterns.

## 9. Risks

- **Capability-property risk:** SFT may install format without competence — the gate
  catches this cheaply; the negative is itself informative.
- **Format overfit despite diversity:** 4 families may still permit family-specific
  shortcuts; the held-out families + suppression ablation are the detectors.
- **Short-form regression:** mitigated by the 15% no-ledger mix and the GSM8K control.
- **Reward hacking in Phase 2:** NOTE-spray guarded by precision·recall scoring.
- **Naturalization corruption:** the rephraser can silently alter facts — training on such
  traces would teach corrupted bookkeeping. Mitigated by the mandatory extract-back +
  equality QC (rejection sampling); QC pass-rate is reported with the dataset.
- **Billing:** Tinker live today; runs should checkpoint frequently (Tier-2 lesson).
