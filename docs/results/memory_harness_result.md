# Memory-Harness Ladder — Steps 1+2 Results (Step 3 pending Tinker access)

**Date:** 2026-06-12
**Provider:** OpenRouter `qwen/qwen3-30b-a3b-instruct-2507`, temperature 0. All ± comparisons
within-provider (Tinker key was revoked server-side mid-program; raw-curl-confirmed; steps 1+2
are sampling-only so they were ported — `scripts/harness_or.py`).
**Predictions under test** (from `rlvr_transfer_postmortem.md`): the d6 capacity cliff is
state-tracking/retrieval, not local computation. (1) An external keyed memory should lift d6 if
retrieval-bound. (2) Enforced decomposition should restore the slip regime p_local^63.

## Results

| condition | d6 | d7 | notes |
|---|---|---|---|
| plain (single call, 8k) | 0.163 (n=160) | 0.087 (n=80) | the cliff |
| **memory harness (zero-shot)** | **0.006** | **0.013** | *worse than plain* |
| **guided (explicit strategy)** | **0.025** (n=160) | **0.062** (n=80) | *also worse than plain* |
| **forced-chunk** (≤4 nodes/round, 600 tok/round, ≤18 rounds) | **0.000** (n=48) | — | never answered; cached 7.5/63 nodes |
| **enforced decomposition** | **0.988** (n=80) | — | local_acc **0.9964** over 63-node chains |

### Step 2 — decisive confirmation of the capacity-cliff mechanism

With the environment carrying state and control (post-order traversal; model answers only
local min/max with its own values propagated), d6 goes **0.16 → 0.99**. Local reliability is
0.9964 across ~5,000 sequential local steps. The cliff is **entirely state/control — local
computation is intact**. (End-to-end 0.988 even exceeds the iid slip prediction 0.798 because
rare local errors mostly hit non-critical branches.)

### Step 1 — zero-shot memory use FAILS: the strategy is the missing ingredient

The harness (notes survive between rounds; reasoning text does not) collapses performance
(0.163 → 0.006). Forensics on transcripts show why:

1. **Transcription, not caching**: the model fills memory with the 64 *leaf values* — data
   already present in the prompt (mean 90 notes vs 63 internal nodes worth caching).
2. **Final-round cram**: it then re-derives long value chains inside a single round anyway —
   reproducing the original overloaded computation, now with extra copy steps.
3. **The same lost-place errors**: e.g. computing `B.B.B = max(−6, −2)` at a node where it is
   MIN's turn — the parity/state error the postmortem identified.

So zero-shot, the model cannot *strategically operate* an external memory: it lacks the
cache-computed-values-bottom-up, work-in-chunks policy. My pre-stated prediction (60–75% that
the harness lifts d6) was **wrong**; the "slip-regime restoration" prediction was **right**.

### Step 1b — GUIDED strategy: knowing the algorithm does not help (0.025, *below* plain)

To bracket Step 3's ceiling — *is the strategy operable once known?* — we handed the model the
exact bottom-up caching algorithm (path notation, MAX-at-even/MIN-at-odd parity rule, "cache
nodes whose children are known, a few per round"). The explicit strategy **fixes the parity
errors** (zero-shot's signature mistake disappears) and yields structurally correct caches —
yet d6 accuracy is **0.025, below plain 0.163**. Why: given the algorithm, the model dumps the
whole 63-node computation in ONE round (mean 1.1 rounds), maximizing local-slip surface instead
of using rounds to checkpoint verified state. **Knowing the procedure is not the bottleneck;
executing 63 bookkeeping steps without losing a value is.** Every "model manages its own state"
condition fails (0.006–0.16); only "environment manages state" succeeds (0.988).

This is a **low ceiling estimate for Step 3** — with one escape hatch the guided probe lacked:
Step 3's RL env caps each round at 1600 tokens, which is *physically too short to cram a d6
tree*, forcing work across rounds with state in notes (the decomp-like regime that scores 0.99).
Whether GRPO can drive the policy into that regime against the model's strong cramming prior is
exactly the open question — now correctly framed as *inducing chunking*, not *teaching the rule*.

### Step 1c — FORCED chunking (the escape hatch, tested by prompt): also fails (0.000)

The remaining hope was that *preventing* cramming would surface the decomp-like regime. We
imposed it directly: ≤600 tokens/round (too short to cram), ≤18 rounds, and a strict protocol
("compute AT MOST 4 ready nodes, save NOTEs, output CONTINUE; only answer once both A and B are
cached"). Result: **d6 = 0.000 (n=48); the model never produced an answer in 18 rounds, caching
only 7.5 of the 63 required nodes**. Forced to chunk, the model cannot run the protocol: it
loses track of which nodes are "ready," recomputes or stalls, and never propagates up to A/B.

So **all four model-managed conditions fail** (plain 0.16, memory 0.006, guided 0.025,
forced-chunk 0.000); the **only** regime that solves d6 is environment-managed control + state
(decomp 0.988). The deficit is not strategy knowledge, not cramming, and not parity — it is the
inability to *reliably execute a long, self-directed bookkeeping loop*. That is a capability
property of in-context multi-step control, and prompting cannot install it.

**Implication for Step 3 (RLVR):** the target behavior is precisely what every prompted variant
fails at. RLVR could in principle reward reaching the answer and thereby select for whatever
control policy works — but the reachable-set evidence (model produces correct d6 answers only as
lottery draws; forced protocol never even completes) means GRPO has almost no positive signal to
climb at d6, and Tiers 1–3 already showed RLVR moves execution reliability ~1pp. The honest
prediction is **small or no uplift**; the RL run is worth executing to confirm, since the env's
token cap creates a path the prompted probes couldn't, but expectations are now low.

## Why this is the ideal setup for Step 3 (memory-strategy RLVR)

The gap is now exactly localized:
- local skill: **0.996** (perfect enough)
- environment-managed strategy: **0.988**
- model-managed strategy: **0.006**

The ONLY missing ingredient is the note-taking/decomposition *policy* — a behavioral,
policy-level object, squarely in RLVR's wheelhouse (unlike the capacity itself, which Tiers
1–3 showed RLVR cannot touch). Headroom is 0.006 → ~0.99. Training signal exists because the
harness permits answering in round 1 (at trained depths d4–d5 the model can fall back to
direct solving, giving mixed groups), and the curriculum (d4→d6) lets the strategy emerge
where direct solving starts failing.

## Step 3 — Memory-strategy RLVR: behavior induced, accuracy payoff pending eval

The multi-round memory GRPO env (`gt_memory_env.py`, `tinker_grpo_mem.py`) trained on
`train_tier4_mem.jsonl` (gametree d4:150/d5:300/d6:350 curriculum; group 8, gpb 24, 1600
tokens/round, ≤6 rounds, kl 0.05). The run survived multiple Tinker outages (402 billing +
recurring transient 404 "Promise not found" on the long late-curriculum episodes) via a
resume-from-checkpoint loop, reaching **batch 20/34 (28 cumulative iters, 62%)** before a
deterministic failure at batch 22 (the d6 episodes grow to 5+ rounds × ~63 notes, apparently
hitting a server-side context/promise limit) plus another billing block stopped it. Checkpoint:
`tinker://2e381db3-…:train:0/sampler_weights/000020`.

### Finding (training phase): RL induces the chunking behavior prompting could not

| batch (curriculum) | reward | in-train correct | rounds | notes |
|---|---|---|---|---|
| 0 (d4) | +0.191 | 0.203 | 3.77 | 15.7 |
| 9 (d5) | +0.045 | 0.062 | 4.26 | 26.8 |
| 18 (d6 onset) | +0.057 | 0.078 | 4.30 | 38.5 |
| 21 (d6) | −0.034 | 0.026 | **5.35** | **63.0** |

The decisive behavioral signal: **notes rise 15.7 → 63.0 and rounds 3.8 → 5.4**, jumping exactly
when the d6 tier begins — under RL pressure the model learns to write ≈ the full 63-node tree
across 5+ rounds, i.e. *cache-bottom-up across rounds*. This is the policy that zero-shot memory
(90 notes but all leaves, 1 effective pass), guided (1.1 rounds), and forced-chunk (stalled at
7.5 notes) all failed to produce. **RLVR moved the policy where prompting could not** — confirming
Step 3's premise that note-taking *strategy* is a trainable, policy-level object.

### Open: does the induced behavior lift accuracy? (held-out eval pending Tinker billing)

The in-training d6 correct rate stays low (0.026) — but that is the curriculum's hardest tier
under sampling temperature, not the clean test. Whether the batch-20 checkpoint beats the
**same-provider base in-harness** on held-out greedy d6/d7 — toward the env-managed 0.988 or
stuck near the probe floor (0.006/0.025/0.000) — requires the Tinker eval suite, which is 402
billing-blocked again. **Early hint is sobering**: low in-train d6 accuracy suggests per-node
slips still compound over 63 self-directed steps even once chunking is induced (consistent with
the low-ceiling prediction). The eval (`tier4_mem_checkpoint.txt` vs base, in/out-of-harness
d6/d7 + transfer) is armed to auto-run and finalize this section on billing restore.

## Reproduce

```bash
# steps 1+2 (OpenRouter, temp 0)
python3 scripts/harness_or.py --mode plain  --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --out data/runs/gt_rlvr/t4or_plain_d6.json
python3 scripts/harness_or.py --mode memory --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --out data/runs/gt_rlvr/t4or_harness_d6.json
python3 scripts/harness_or.py --mode decomp --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --limit 80 --out data/runs/gt_rlvr/t4or_decomp_d6.json
# (same for eval_t3_d7.jsonl)  # step 3: see tinker_grpo_mem.py
```

Artifacts: `data/runs/gt_rlvr/t4or_{plain,harness}_{d6,d7}.json`, `t4or_decomp_d6.json`.
