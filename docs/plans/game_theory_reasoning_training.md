# Plan: can game-theory–synthesized data train better long-depth reasoners?

**Hypothesis.** Game-theoretic structures are an unusually good *source* of
training data for long-depth reasoning because they uniquely combine three
properties most synthetic-reasoning corpora lack at once:

1. **Tunable depth.** Backward induction, iterated dominance, and level-k belief
   recursion each have a single parameter (game length / elimination rounds /
   belief level) that *is* the required reasoning depth. → a curriculum dial.
2. **Programmatic verifiability.** The optimal action / equilibrium value is
   exactly computable by a solver. → a correctness reward for RLVR or a filter
   for rejection-sampling SFT, with no human labels and no answer-key leakage.
3. **Infinite, contamination-free, diversely-framable.** We can generate
   unlimited instances, and the existing framework re-frames one deep structure
   into many surface forms (negotiation / board-game / military / abstract), so
   the model must learn the *reasoning*, not a template.

The research question: does training on this data make a model a **better
long-depth reasoner that generalizes** — to deeper instances than trained on
(extrapolation), to other game families, and to *external* reasoning benchmarks?
Per our standard, a "better reasoner" claim must hold across **≥3 benchmarks**
with the same checkpoint (a within-distribution gain alone is overfitting — this
is exactly the trap the steering experiment fell into).

## Task families (verifiable, depth-controllable)

Implemented in `game_theory_llm/reasoning/gametree.py` (solver unit-verified):
- **Minimax over game trees** (depth = plies of lookahead) — the canonical
  long-depth task; answer = root value under optimal play; verifier = backward
  induction. *Primary.*
- **Level-k beauty contest** (depth = belief-recursion levels).
- **Iterated dominance** (depth = elimination rounds). *(scaffolded; extend)*

Further candidates (brainstorm, by reasoning type):
backward-induction bargaining (Rubinstein finite-horizon), Nim/Sprague-Grundy
(combinatorial), centipede, repeated-game equilibria, Colonel Blotto allocations,
auction/mechanism reasoning, sequential-elimination tournaments. Each has a
closed-form or polynomial solver = verifier.

## Phase 0 — Feasibility (runnable now, no Tinker)  *[in progress]*
- ✅ Verifiable generator + solver (`gametree.py`, 200-tree brute-force check).
- 🔄 **Depth-difficulty baseline** (`scripts/run_gametree_baseline.py`): E4B
  unsteered accuracy vs depth. *Premise check: accuracy must decay with depth
  (headroom to train into) and the generate→verify loop must be clean.*
- Next: a small held-out eval suite — in-distribution depths, **deeper** depths
  (extrapolation), a **held-out family**, and the external set (GSM8k, MMLU, BBH).

## Phase 1 — Data synthesis (no Tinker)
- Generate a depth-curriculum corpus (e.g. depths 1–6, balanced families, diverse
  framings), each with: prompt, verifiable answer, and a **gold backward-induction
  CoT** (the solver can emit a worked trace) for SFT targets.
- Build a **rejection-sampling (STaR-style) set**: sample the base model's own CoT,
  keep only solutions that reach the verified answer → high-quality self-generated
  SFT data that stays on-distribution for the model.

## Phase 2 — Training (needs Tinker API)
Three options, increasing power; all use the verifier:
1. **Curriculum SFT** on gold/accepted CoT, easy→hard depths.
2. **Rejection-sampling SFT (STaR)**: iterate generate→verify→keep→fine-tune.
3. **RLVR**: RL with the verifier as a 0/1 reward (Tinker supports custom-reward RL).
Start with (2) — cheapest, strong, no reward-hacking surface.

## Phase 3 — Validation (the actual answer to the question)
Train on a *subset* of depths/families; evaluate the same checkpoint on:
- **In-distribution depth** (sanity).
- **Depth extrapolation** — deeper trees than trained on (the real "long-depth" test).
- **Held-out game family** (e.g. train trees, test bargaining/Nim).
- **External reasoning benchmarks** — **GSM8k, MMLU, BBH** (≥3) — to test whether
  game-reasoning training transfers to *general* reasoning, scored as
  accuracy-among-parsed.
Only a checkpoint that improves depth-extrapolation **and** holds/improves on the
external set counts as "a better long-depth reasoner." Guardrails: no degradation
on the capability set (HumanEval/MMLU), and a contamination check (fresh seeds).

## Design review (methodology skills)

Sharpened with `critical-depth-threshold-sweep` and
`counterfactual-component-ablation-controlled-fixing`:

1. **Depth as the swept axis, with the token-budget confound controlled.** Sweep
   reasoning depth at fine granularity with multiple seeds and 95% CIs, and
   locate the *critical depth* where accuracy collapses. The sweep skill's
   warning ("LR not re-tuned per depth → false plateau") maps directly to a real
   confound here: **deep trees need more output tokens**, so a fixed
   `max_new_tokens` conflates "can't reason that deep" with "ran out of tokens."
   Mitigation: scale the generation budget with tree size and **report
   truncation/parse rate per depth**, so accuracy decay is attributed to
   reasoning, not truncation. (Our first baseline used a fixed 1024-token budget;
   the real sweep fixes this and reports the per-depth truncation rate.)

2. **A counterfactual control fine-tune to isolate the game-theory contribution.**
   A fine-tune that improves reasoning could be driven by (a) game-theoretic
   reasoning content or (b) generic CoT / long-output / format practice. Per the
   controlled-fixing pattern, we hold the game-theoretic-reasoning component
   *fixed/removed* while keeping everything else identical: a **control fine-tune
   on structure-matched but reasoning-trivial data** (depth-1 / shuffled trees in
   the same surface format and CoT length, or generic CoT) under the same recipe
   and token count. The game-theory gain counts only to the extent it *exceeds*
   this control on held-out reasoning. (Diagnostic, not prescriptive.)

3. **Train vs. extrapolation split.** Fit on a band of depths (e.g. {2,3,4}) and
   evaluate both in-band and on **deeper** depths {5,6,7,8} — depth extrapolation
   is the actual "long-depth reasoner" test, distinct from in-distribution gains.

4. **Multi-axis ablation.** Vary one axis at a time — train-depth band, game
   family, SFT-set size, training method (SFT vs rejection-sampling vs RLVR) — so
   each effect is attributable.

5. **Validation bar (unchanged):** the same checkpoint must improve depth
   extrapolation **and** hold/improve on ≥3 external benchmarks (GSM8k, MMLU,
   BBH, accuracy-among-parsed), with no capability regression — else it is
   structure-specific, not a general long-depth gain.

## What we learn either way
- **If it transfers:** game theory is a scalable, verifiable curriculum for
  long-depth reasoning — a free, continual-learning-ready data engine.
- **If it doesn't:** we'll have shown (as with steering) that the gain is
  structure-specific — still a clean, publishable negative with a strong method.

## Open dependency
Phase 2 needs **Tinker API access** (a `TINKER_API_KEY` / SDK) — not currently
configured in this repo. Phases 0–1 (data engine + baselines + SFT-set
construction) are fully runnable now.
