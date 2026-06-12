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

**Step 3 status: BLOCKED on Tinker credential** (server-side key revocation; raw curl to
`/api/v1/auth/token` returns 401 for the unchanged key that ran Tiers 1–3). Training env
(`gt_memory_env.py`), launcher (`tinker_grpo_mem.py`), and curriculum
(`train_tier4_mem.jsonl`, d4:150/d5:300/d6:350) are built, validated, and committed; the run
fires on the next successful auth probe.

## Reproduce

```bash
# steps 1+2 (OpenRouter, temp 0)
python3 scripts/harness_or.py --mode plain  --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --out data/runs/gt_rlvr/t4or_plain_d6.json
python3 scripts/harness_or.py --mode memory --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --out data/runs/gt_rlvr/t4or_harness_d6.json
python3 scripts/harness_or.py --mode decomp --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --limit 80 --out data/runs/gt_rlvr/t4or_decomp_d6.json
# (same for eval_t3_d7.jsonl)  # step 3: see tinker_grpo_mem.py
```

Artifacts: `data/runs/gt_rlvr/t4or_{plain,harness}_{d6,d7}.json`, `t4or_decomp_d6.json`.
