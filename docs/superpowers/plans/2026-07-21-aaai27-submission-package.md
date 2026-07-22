# AAAI-27 Submission Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the complete AAAI-27 main-track package today (2026-07-21): `aaai27.tex` (7 content pages + ≤2 ref pages, AAAI author kit), `aaai27_supplementary.tex`, reproducibility-checklist draft, anonymized code zip, and a verified five-engine game-mechanics audit.

**Architecture:** Two independent workstreams. Workstream A (Tasks 1–2) fans out six read-only audit subagents over the game engines and shared play infra in the steering worktree, then adversarially verifies findings into one report. Workstream B (Tasks 3–8) ports the frozen NeurIPS draft to the AAAI kit on a new `aaai27` branch of the paper repo, builds the supplementary, checklist, and code zip. Task 9 is the QA gate over everything. A starts before B and runs concurrently.

**Tech Stack:** LaTeX (pdflatex/latexmk, AAAI-27 author kit, natbib), bash, python3, git. Paper repo: `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/-NEURIPS-2026-Framing-The-Game` (own git repo). Steering worktree: `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering`.

## Global Constraints

- Content is FROZEN: no science changes, no new experiments, no rewording of claims beyond compression.
- Main paper: **7 content pages max; pages 8–9 references only**; AAAI two-column US Letter; Type 1/TrueType fonts; anonymous; no acknowledgements.
- The AAAI main document is named `aaai27.tex`; supplementary is `aaai27_supplementary.tex`; both live on branch `aaai27` of the paper repo; `main.tex` stays untouched.
- Pre-agreed cut order if over 7 pages: (a) compress Related Work, (b) trim Results prose restating figure content, (c) move agreement/consistency detail to supplementary. Nothing load-bearing for a claim leaves the main text.
- Audit verdict tiers: clean / bug-not-affecting-reported-numbers (fix + note) / bug-affecting-reported-numbers → escalate to user immediately with re-run cost estimate.
- Audit subagents are READ-ONLY (no file modifications); parallel dispatch is safe.
- The tier-5 GRPO v2 background run is untouched and out of scope.
- Abstract registration on OpenReview is the user's own action tonight (not a task).

---

### Task 1: Dispatch six parallel game-mechanics audit subagents

**Files:**
- Create: `/private/tmp/claude-501/-Users-isaacrobinson-Documents-Contextual-Game-Theoretic-LLM-Evaluations/8fc29533-7c9f-40f0-a5b7-ec4471eaf7cd/scratchpad/audit-<target>-findings.md` (one per subagent; `<target>` ∈ `diplomacy`, `secret-hitler`, `onw`, `monopoly`, `risk`, `infra`)
- No repo files modified.

**Interfaces:**
- Produces: six findings files consumed by Task 2. Each finding has: `severity` (Critical/Important/Minor), `file:line`, `claim`, `evidence`, `affects_reported_numbers` (yes/no/unsure + which number).

- [ ] **Step 1: Dispatch all six audits in one parallel batch (read-only, Explore-style prompts to general-purpose agents).** Use this prompt template, substituting `<ENGINE_FILE>`, `<TEST_FILE>`, `<GAME_NAME>`, `<OUT_FILE>`:

```
You are auditing the correctness of a game engine used to produce published
research numbers. READ-ONLY audit: do not modify any file.

Engine: /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering/<ENGINE_FILE>
Tests:  /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering/<TEST_FILE>
Shared infra (read for context as needed, same dir): game_theory_llm/play/{base,runner,messaging,alliances,metrics}.py and game_theory_llm/play/maps/

Audit <GAME_NAME> on exactly four dimensions:
1. RULES FIDELITY - the implementation matches its own documented (lite)
   ruleset: turn order, legal-action generation, action resolution, win
   conditions. The module docstring/comments state the intended rules; flag
   any divergence between stated rules and code.
2. HIDDEN-INFORMATION INTEGRITY - trace every string that reaches another
   seat's prompt (via runner/messaging). Flag any leak of private state:
   hidden roles, private messages, hidden cards/holdings, deck contents.
3. METRIC CORRECTNESS - win attribution and any values fed to the LLM judge
   or metrics module are computed from the correct state. Also verify: when
   a model's move fails to parse, the fallback move is drawn UNIFORMLY from
   the LEGAL move set (the paper claims "unbiased random legal moves").
4. TEST-COVERAGE GAPS - which of the above properties have no test.

For each finding, write to <OUT_FILE> in this exact format:
## Finding N
- severity: Critical | Important | Minor
- location: file:line
- claim: <one sentence>
- evidence: <code excerpt + why it is wrong>
- affects_reported_numbers: yes/no/unsure - <which number if yes/unsure>

End the file with '## Summary' and a per-dimension verdict (clean/issues).
If a dimension is clean, say so explicitly. Return only the path of the file
you wrote and a one-line count of findings by severity.
```

Dispatch table:

| target | ENGINE_FILE | TEST_FILE | GAME_NAME |
|---|---|---|---|
| diplomacy | `game_theory_llm/play/games/diplomacy_lite.py` | `tests/play/test_diplomacy.py` (+`test_maps_diplomacy.py`) | Diplomacy-lite |
| secret-hitler | `game_theory_llm/play/games/secret_hitler.py` | `tests/play/test_sh.py` | Secret Hitler |
| onw | `game_theory_llm/play/games/one_night_werewolf.py` | `tests/play/test_onw.py` (+`test_maps_onw.py`) | One Night Werewolf |
| monopoly | `game_theory_llm/play/games/monopoly_lite.py` | `tests/play/test_monopoly.py` | Monopoly-lite |
| risk | `game_theory_llm/play/games/risk_lite.py` | `tests/play/test_risk.py` (+`test_maps_risk.py`) | Risk-lite |
| infra | `game_theory_llm/play/{runner,messaging,alliances,metrics,viewer}.py` | `tests/play/{test_runner_v2_schema,test_messaging,test_alliances,test_metrics}.py` | shared play infrastructure (audit dimensions 2 and 3 only, across all games) |

- [ ] **Step 2: Verify all six dispatches are running**, then proceed to Task 3 (Workstream B) while they work. Do NOT block on audit completion here.

### Task 2: Verify audit findings and assemble the audit report

**Files:**
- Create: `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering/docs/results/game_mechanics_audit.md`
- Read: the six findings files from Task 1.

**Interfaces:**
- Consumes: Task 1 findings files.
- Produces: committed audit report; escalation decision input for the user.

- [ ] **Step 1: Collect the six findings files** (when Task 1 agents report back; this task runs after Tasks 3–5 if audits are still in flight).
- [ ] **Step 2: Adversarially verify every Critical/Important finding**: for each, dispatch one verifier subagent (or verify inline by reading the code) with the prompt "Try to REFUTE this finding: <finding>. Read <file> around <line>. Default to refuted if the code is actually correct." Drop refuted findings; keep confirmed ones.
- [ ] **Step 3: Write the report** `docs/results/game_mechanics_audit.md` in the steering worktree with: date, scope (5 engines + infra, line counts), method (parallel read-only audits + adversarial verification), per-engine verdict table (rules / hidden-info / metrics / coverage), confirmed findings with file:line, and an explicit section "Findings affecting reported numbers" (empty or not).
- [ ] **Step 4: Escalate if needed.** If any confirmed finding has `affects_reported_numbers: yes` for ONW/Secret Hitler (paper) or Monopoly/Diplomacy/Risk (steering report), STOP and present to the user with a re-run cost estimate before continuing the paper package.
- [ ] **Step 5: Commit** (in the steering worktree):

```bash
cd /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering
git add docs/results/game_mechanics_audit.md
git commit -m "audit: full game-mechanics audit of five play engines + shared infra"
```

### Task 3: Create `aaai27` branch and a compiling AAAI-kit skeleton

**Files:**
- Create (paper repo): `aaai27.tex`, `aaai27.sty`, `aaai27.bst` (from the official kit)
- Repo: `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/-NEURIPS-2026-Framing-The-Game`

**Interfaces:**
- Produces: branch `aaai27`; `aaai27.tex` skeleton that compiles under the official kit with title + abstract; kit style files in repo root. Tasks 4–6 build on this.

- [ ] **Step 1: Branch:**

```bash
cd "/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/-NEURIPS-2026-Framing-The-Game"
git checkout -b aaai27
```

- [ ] **Step 2: Download the official AAAI-27 author kit.** Fetch https://aaai.org/authorkit27/ and follow its download link (a zip, typically `AuthorKit27.zip`). Unzip into a scratch dir; copy `aaai27.sty` and `aaai27.bst` (exact names per kit) into the repo root. If the kit page is unreachable, STOP and report — do not substitute a prior year's kit silently.
- [ ] **Step 3: Create `aaai27.tex` skeleton** using the kit's own template preamble (authoritative over the sketch below), with our title/abstract wired in:

```latex
\documentclass[letterpaper]{article}
\usepackage[submission]{aaai27}      % kit option per its README (submission/anonymous mode)
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,multirow}
\urlstyle{rm}
\def\UrlFont{\rm}
\frenchspacing
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}
\title{Framing the Game: How Context Shapes LLM Decision-Making}
\author{Anonymous submission}        % per kit's anonymous-submission instructions
\begin{document}
\maketitle
\begin{abstract}
% paste abstract verbatim from main.tex lines 61-63
\end{abstract}
% body arrives in Task 4
\bibliographystyle{aaai27}
\bibliography{references}
\end{document}
```

Notes: the kit forbids some packages (historically hyperref, geometry changes); follow the kit README. `main.tex` uses `\citep/\citet` (natbib) — keep natbib loading per the kit's prescribed method (`\usepackage[round]{natbib}` only if the kit template does so; otherwise use the kit's citation setup and adapt commands mechanically).

- [ ] **Step 4: Compile and verify:**

```bash
latexmk -pdf -interaction=nonstopmode aaai27.tex
```

Expected: `aaai27.pdf` builds, two-column, title + abstract, zero errors.

- [ ] **Step 5: Commit:**

```bash
git add aaai27.tex aaai27.sty aaai27.bst
git commit -m "aaai27: kit skeleton compiles (title + abstract)"
```

### Task 4: Port the full body into `aaai27.tex`

**Files:**
- Modify (paper repo): `aaai27.tex`
- Read: `main.tex` (body = lines 66–279), `mechanistic_analysis.tex`, `activation_steering.tex`

**Interfaces:**
- Consumes: Task 3 skeleton.
- Produces: complete-body `aaai27.tex` (likely over-length; Task 5 compresses). Section labels/refs preserved so `\Cref`/`\ref` targets keep working (replace `\Cref` with `\ref` + manual "Section"/"Figure" text if the kit forbids cleveref).

- [ ] **Step 1: Port body sections in order**, verbatim except mechanical adaptations: Introduction; Related Work; Methodology; Results (all subsections); the *content* of `mechanistic_analysis.tex` and `activation_steering.tex` inlined (keep using `\input` only if both files need no kit-specific changes; otherwise inline); Discussion; Conclusion. EXCLUDE: NeurIPS Reproducibility Statement, LLM-Usage section, all appendix sections (they go to supplementary in Task 6), `\input{neurips_checklist}`.
- [ ] **Step 2: Figure pass.** List figures: `grep -n includegraphics main.tex mechanistic_analysis.tex activation_steering.tex`. Headline multi-panel figures (behavioral overview, dose–response, gameplay X-cross) become `\begin{figure*}` spanning both columns; single-panel figures stay column-width `\begin{figure}` with `width=\columnwidth`. All graphics files already exist in `ResultsFigures/` or repo root — do not regenerate.
- [ ] **Step 3: Compile; fix all errors and undefined references** (missing labels must be resolved, not deleted — if a `\ref` targets an appendix section, repoint the text to "the supplementary material").
- [ ] **Step 4: Record page count:**

```bash
latexmk -pdf -interaction=nonstopmode aaai27.tex && mdls -name kMDItemNumberOfPages aaai27.pdf
```

Expected: compiles clean; note where content ends and references begin.

- [ ] **Step 5: Commit:**

```bash
git add aaai27.tex
git commit -m "aaai27: full body ported (pre-compression, N pages)"
```

### Task 5: Compress to 7 content pages + ≤2 reference pages

**Files:**
- Modify (paper repo): `aaai27.tex`

**Interfaces:**
- Consumes: Task 4 output and its page count.
- Produces: length-compliant `aaai27.pdf`; a list of every passage moved to supplementary (consumed by Task 6).

- [ ] **Step 1: Apply the pre-agreed cut order until content fits 7 pages** — (a) compress Related Work to ~2/3 length by merging citation sentences; (b) trim Results prose that restates figure content; (c) move agreement/consistency detail to supplementary (add "see supplementary material" pointers). Rules: numbers, claims, and statistical tests in the main text NEVER change or disappear; compression is prose-only. Keep a running list `MOVED_TO_SUPP.md` (repo root, untracked scratch) of every passage moved.
- [ ] **Step 2: Verify references fit pages 8–9.** If over, switch to the kit's condensed bib style only if the kit provides one; otherwise prune duplicate/arXiv-superseded entries in `references.bib` (never prune cited works).
- [ ] **Step 3: Final length check:**

```bash
latexmk -pdf -interaction=nonstopmode aaai27.tex && mdls -name kMDItemNumberOfPages aaai27.pdf
```

Expected: total ≤ 9 pages; visually confirm content ends on page 7.

- [ ] **Step 4: Commit:**

```bash
git add aaai27.tex references.bib
git commit -m "aaai27: compressed to 7+2 page budget"
```

### Task 6: Build `aaai27_supplementary.tex`

**Files:**
- Create (paper repo): `aaai27_supplementary.tex`
- Read: `main.tex` appendix (lines 281–607), `MOVED_TO_SUPP.md` from Task 5

**Interfaces:**
- Consumes: Task 3 kit files, Task 5 moved-passage list.
- Produces: compiled `aaai27_supplementary.pdf`.

- [ ] **Step 1: Create the document** with the same kit preamble as `aaai27.tex` (same class/style, anonymous), title "Supplementary Material: Framing the Game". Port ALL appendix sections from `main.tex`: Game Suite; Generation Process + Example Vignette; More Vignette Examples; Vignette Validation; API Details; XGBoost Details; Contrast Statistics; Decision Consistency/Agreement; Per-Model Heatmaps; Temporal Trends; Layer-Wise Probe; MoralBench Breakdown — plus every passage listed in `MOVED_TO_SUPP.md`.
- [ ] **Step 2: Compile clean:**

```bash
latexmk -pdf -interaction=nonstopmode aaai27_supplementary.tex
```

- [ ] **Step 3: Commit:**

```bash
git add aaai27_supplementary.tex
git commit -m "aaai27: supplementary document (full appendix)"
```

### Task 7: Draft the AAAI reproducibility checklist

**Files:**
- Create (paper repo): `aaai27_repro_checklist.md`
- Read: `neurips_checklist.tex`

**Interfaces:**
- Produces: draft answers the user pastes into the OpenReview form.

- [ ] **Step 1: Fetch the AAAI-27 reproducibility checklist questions** (linked from https://aaai.org/conference/aaai/aaai-27/submission-instructions/ or the author kit). Copy the exact question list into `aaai27_repro_checklist.md`.
- [ ] **Step 2: Draft an answer per question** (yes/no/NA + one-line justification) derived from `neurips_checklist.tex` facts: code released in supplement, procedurally generated data, API models named with exact IDs, statistical tests specified (Fisher, Bonferroni), compute described, no human subjects.
- [ ] **Step 3: Commit:**

```bash
git add aaai27_repro_checklist.md
git commit -m "aaai27: reproducibility checklist draft"
```

### Task 8: Anonymized code/data supplement zip

**Files:**
- Create: `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/-NEURIPS-2026-Framing-The-Game/aaai27_code_supplement.zip` (git-ignored; add `aaai27_code_supplement.zip` to the paper repo `.gitignore`)

**Interfaces:**
- Consumes: main repo `game_theory_llm/` + `scripts/` + paper repo `scripts/`.
- Produces: the zip uploaded with supplementary material.

- [ ] **Step 1: Stage a clean tree in the scratchpad:**

```bash
S=/private/tmp/claude-501/-Users-isaacrobinson-Documents-Contextual-Game-Theoretic-LLM-Evaluations/8fc29533-7c9f-40f0-a5b7-ec4471eaf7cd/scratchpad/code_supp
mkdir -p "$S" && cd /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations
rsync -a --exclude '.git' --exclude '__pycache__' --exclude '.env*' --exclude 'data/' game_theory_llm scripts tests pyproject.toml README.md "$S/" 2>/dev/null || true
rsync -a "-NEURIPS-2026-Framing-The-Game/scripts/" "$S/paper_figure_scripts/"
```

- [ ] **Step 2: Anonymization scrub — must return zero hits before zipping:**

```bash
grep -rniE "isaac|robinson|burden|robinsonaisaac|@gmail|@cam\.ac\.uk|github\.com/[A-Za-z]" "$S" | grep -v ".bib" ; echo "exit=$?"
```

Expected: `exit=1` (no matches). Fix any hit by redacting the line, then re-run. Also verify no `.env`, no API keys: `grep -rn "sk-or-\|OPENROUTER_API_KEY=" "$S"` must be empty.

- [ ] **Step 3: Zip and place:**

```bash
cd "$S" && zip -qr aaai27_code_supplement.zip . && mv aaai27_code_supplement.zip "/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/-NEURIPS-2026-Framing-The-Game/"
```

- [ ] **Step 4: Commit the .gitignore change only:**

```bash
cd "/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/-NEURIPS-2026-Framing-The-Game"
echo "aaai27_code_supplement.zip" >> .gitignore && git add .gitignore && git commit -m "aaai27: ignore code supplement zip"
```

### Task 9: QA gates and final package report

**Files:**
- Read: `aaai27.pdf`, `aaai27_supplementary.pdf`, `aaai27_code_supplement.zip`, audit report.

**Interfaces:**
- Consumes: everything above.
- Produces: the done/not-done verdict and the final summary to the user.

- [ ] **Step 1: Page/format gates:**

```bash
cd "/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/-NEURIPS-2026-Framing-The-Game"
mdls -name kMDItemNumberOfPages aaai27.pdf          # expect <= 9
pdffonts aaai27.pdf | awk 'NR>2 && $(NF-4)=="no"'    # expect no output (all embedded)
```

- [ ] **Step 2: Anonymity gate on the PDF text:**

```bash
pdftotext aaai27.pdf - | grep -niE "isaac|robinson|burden|cambridge|@" ; echo "exit=$?"
```

Expected `exit=1` (the bib page may legitimately contain author names of CITED works — only OUR names/affiliation are violations; adjudicate hits by eye).

- [ ] **Step 3: Reference integrity:** `grep -c "??" aaai27.log` style check — run `latexmk -pdf` once more and confirm zero "undefined references" warnings in the log.
- [ ] **Step 4: Visual spot-check:** open `aaai27.pdf`; confirm two-column layout, figures legible at print size, content ends page 7, refs end ≤ page 9. Same spot-check on supplementary.
- [ ] **Step 5: Confirm audit closure:** Task 2 report committed and no unresolved escalations.
- [ ] **Step 6: Final report to user:** deliverable paths, page counts, audit verdict summary, and the exact remaining human steps (register abstract tonight; upload paper by Jul 28; upload supplementary zip + checklist by Jul 31).

---

## Self-Review Notes

- Spec coverage: aaai27.tex (T3–5), supplementary (T6), checklist (T7), code zip (T8), audit (T1–2), QA gates (T9), naming and branch rules in Global Constraints — all spec items mapped.
- Deliberate deviation from strict TDD: LaTeX tasks use compile-and-page-count as their test cycle; audit tasks use adversarial verification as theirs. No unit-test steps exist because no production code changes.
- Type consistency: file names `aaai27.tex` / `aaai27_supplementary.tex` / `aaai27_repro_checklist.md` / `aaai27_code_supplement.zip` used identically across tasks.
