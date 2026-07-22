# AAAI-27 Submission Package — Design

**Date:** 2026-07-21 (all work completes TODAY)
**Owner:** Isaac Robinson (authors locked: Isaac Robinson, John Burden)

## Goal

Produce a complete, submission-ready AAAI-27 main-track package from the existing
NeurIPS draft, with zero science changes, plus a full mechanics audit of the five
game engines backing the multi-agent results. Everything finishes today,
2026-07-21; the OpenReview uploads then happen well ahead of the official
deadlines (abstract Jul 21 AoE — tonight, user does this; full paper Jul 28;
supplementary/code Jul 31).

## Deliverables

1. **`aaai27.tex`** — the AAAI main paper, on a new `aaai27` branch of the paper
   repo `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/-NEURIPS-2026-Framing-The-Game`.
   - Fresh file; `main.tex` (NeurIPS version) stays untouched.
   - Official AAAI-27 author kit (https://aaai.org/authorkit27/): two-column,
     US Letter, Type 1/TrueType fonts only.
   - Body = current Introduction → Conclusion including `mechanistic_analysis.tex`
     and `activation_steering.tex` content. **7 content pages max; references on
     pages 8–9 only.**
   - NeurIPS checklist and NeurIPS-specific sections (LLM-usage statement,
     NeurIPS reproducibility statement) removed from the PDF; their facts feed
     the AAAI reproducibility checklist instead.
   - Anonymous (already is); no acknowledgements.
   - Pre-agreed cut order if over 7 pages: (a) compress Related Work,
     (b) trim Results prose restating figure content, (c) move
     agreement/consistency detail to supplementary. Nothing load-bearing for a
     claim leaves the main text.
   - Figure pass: headline figures become `figure*` (two-column span); others
     get column-width versions.

2. **`aaai27_supplementary.tex`** — separate AAAI-format document holding the
   current appendix: game suite, generation process + vignettes, vignette
   validation, API details, XGBoost details, contrast statistics,
   agreement/consistency, per-model heatmaps, temporal trends, layer-probe
   curves, MoralBench breakdown.

3. **AAAI reproducibility checklist draft** — answers drafted from the existing
   `neurips_checklist.tex` content, saved as `aaai27_repro_checklist.md` in the
   paper repo (final entry happens in the OpenReview form).

4. **Code/data supplement zip** — anonymized archive containing the main repo's
   `game_theory_llm/` package, the `scripts/` used for the paper's results, and
   the paper repo's figure-generation scripts; scrub grep for names, emails,
   identifying URLs/paths before zipping.

5. **Game-mechanics audit report** —
   `.worktrees/steering/docs/results/game_mechanics_audit.md`.

## Game-Mechanics Audit

**Scope:** all five engines in
`.worktrees/steering/game_theory_llm/play/games/` — `diplomacy_lite.py` (1504),
`secret_hitler.py` (1090), `one_night_werewolf.py` (1063), `monopoly_lite.py`
(984), `risk_lite.py` (796) — plus shared play infrastructure that affects game
correctness (`runner.py`, `messaging.py`, `alliances.py`, `metrics.py`, `maps/`).

**Method:** one review subagent per engine, all five in parallel; a sixth covers
shared infrastructure. Each audits:
1. **Rules fidelity** — implementation matches its documented (lite) ruleset:
   turn order, legal-action generation, resolution logic, win conditions.
2. **Hidden-information integrity** — no private state leaks into other seats'
   prompts (roles in ONW/Secret Hitler; private messages/alliances in
   Diplomacy/Risk; opponent holdings in Monopoly).
3. **Metric correctness** — win attribution and LLM-judge inputs computed from
   the right state; verify the paper's explicit claim that parse-fallback moves
   are *unbiased random legal moves*.
4. **Test-coverage gaps** for any of the above.

**Verification:** every finding is adversarially verified before it counts.

**Verdict tiers per engine:** clean / bug-not-affecting-reported-numbers (fix +
note in audit report) / **bug-affecting-reported-numbers** → escalate to user
immediately with a re-run cost estimate (ONW + Secret Hitler numbers are in the
submission; Monopoly/Diplomacy/Risk numbers are in the steering report).

## QA Gates (before package is called done)

- Page-limit check on compiled `aaai27.pdf` (7 content + ≤2 refs).
- `pdffonts`: all fonts embedded, Type 1/TrueType.
- Anonymization grep over PDF text and code zip: author names, emails,
  institution, repo URLs.
- Figure legibility at print size (column-width figures readable).
- References compile without missing-citation warnings.
- AAAI style compliance: no page-geometry hacks, kit-compliant captions/sections.

## Execution Notes

- Same-day parallelism: audit subagents run concurrently with the LaTeX port
  (different repos, no file conflicts). Audit findings must land before the
  package is declared done.
- The tier-5 GRPO v2 background run continues untouched; it is a separate
  project and not part of this package.
- Abstract registration on OpenReview is the user's action tonight (package
  already delivered in-session: title, abstract verbatim from draft, primary
  area ML–interpretability, secondary NLP, keywords, locked author list).

## Out of Scope

- Any new experiments or content changes (straight port; strengthening options
  were explicitly declined).
- AI Alignment special-track switch — revisit only if its CFP posts with
  meaningfully later dates; this package carries over as-is.
