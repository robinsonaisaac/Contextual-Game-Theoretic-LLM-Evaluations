# Multi-Game Extension Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the experimental framework from PD-only to seven 2×2 games covering all distinct strategic equilibrium structures (PD, Stag Hunt, Chicken, Matching Pennies, Harmony, Deadlock, Battle of the Sexes), and align the generator's API with the `ExperimentConfig` schema.

**Architecture:** Replace `_PURE_COORDINATION` with `_DEADLOCK` in `games.py`. Add a `framing_hint` field to `GameConfig` for game-specific narrative guidance. Refactor `generator.create_query` to accept a `GameConfig` (carrying matrix + semantic labels + framing hint) and align its signature with `ExperimentConfig` (rename `world_type` → `observability`, add `power_dynamic`). Propagate `game.id` onto `Story.game_type` so existing analysis (already aware of `game_type`) works without modification. Add three new analysis helpers and one visualization for cross-game claims.

**Tech Stack:** Python 3.9, pytest, pandas, matplotlib, dataclasses.

**Design doc:** `docs/plans/2026-05-02-multi-game-extension-design.md`

---

## Phase 1: Swap Pure Coordination → Deadlock in games.py

### Task 1: Update test_games.py to expect Deadlock instead of Pure Coordination

**Files:**
- Modify: `tests/test_games.py:20-25` (expected_game_ids)
- Modify: `tests/test_games.py:95-99` (test_pure_coordination_R_gt_P_gt_T_eq_S → test_deadlock_T_gt_P_gt_R_gt_S)

**Step 1: Edit `tests/test_games.py:20-25`** to swap `"pure_coordination"` for `"deadlock"`:

```python
def test_expected_game_ids(self):
    expected = {
        "prisoners_dilemma", "stag_hunt", "chicken", "deadlock",
        "harmony", "battle_of_the_sexes", "matching_pennies",
    }
    assert set(GAME_REGISTRY.keys()) == expected
```

**Step 2: Replace `test_pure_coordination_R_gt_P_gt_T_eq_S`** (lines 95-99) with:

```python
def test_deadlock_T_gt_P_gt_R_gt_S(self):
    R, S, T, P = self._payoffs("deadlock")
    assert T > P > R > S, f"Deadlock ordinal violated: T={T}, P={P}, R={R}, S={S}"

def test_deadlock_dominant_NE_is_pareto_optimal(self):
    """In Deadlock (vs PD), mutual defection is mutually preferred."""
    m = GAME_REGISTRY["deadlock"].matrix.matrix
    # BB Pareto-dominates AA for both players
    assert m[3][0] > m[0][0], "Deadlock: agent 1 prefers BB over AA"
    assert m[3][1] > m[0][1], "Deadlock: agent 2 prefers BB over AA"
```

**Step 3: Run tests to verify they fail with the expected reason**

Run: `python3 -m pytest tests/test_games.py -v`
Expected: `test_expected_game_ids` and `test_deadlock_*` fail because `deadlock` not in registry yet.

**Step 4: Commit**

```bash
git add tests/test_games.py
git commit -m "Update test_games.py to expect Deadlock instead of Pure Coordination"
```

---

### Task 2: Replace `_PURE_COORDINATION` with `_DEADLOCK` in games.py

**Files:**
- Modify: `game_theory_llm/games.py:113-122` (delete `_PURE_COORDINATION`)
- Modify: `game_theory_llm/games.py:9-29` (update docstring table)
- Modify: `game_theory_llm/games.py:162-173` (update GAME_REGISTRY)

**Step 1: In `games.py`**, delete the `_PURE_COORDINATION` definition (lines 113-122) and replace with `_DEADLOCK`:

```python
_DEADLOCK = GameConfig(
    id="deadlock",
    name="Deadlock",
    matrix=PayoffMatrix([(1, 1), (0, 3), (3, 0), (2, 2)]),
    label_a="Cooperate",
    label_b="Defect",
    nash_equilibria=("BB",),
    pareto_optimal="BB",
    description="T>P>R>S: defection dominates AND is mutually preferred — no dilemma (PD's control)",
)
```

**Step 2: Update the docstring table** at lines 11-28 to replace the Pure Coordination row with:

```
| Deadlock            | BB          | no dilemma       | Cooperate / Defect |
|                     |             | (PD control)     |                    |
```

**Step 3: Update `GAME_REGISTRY`** (lines 162-173) to use `_DEADLOCK` instead of `_PURE_COORDINATION`:

```python
GAME_REGISTRY: Dict[str, GameConfig] = {
    g.id: g
    for g in (
        _PRISONERS_DILEMMA,
        _STAG_HUNT,
        _CHICKEN,
        _MATCHING_PENNIES,
        _HARMONY,
        _DEADLOCK,
        _BATTLE_OF_THE_SEXES,
    )
}
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_games.py -v`
Expected: all tests pass.

**Step 5: Commit**

```bash
git add game_theory_llm/games.py
git commit -m "Replace Pure Coordination with Deadlock in game registry"
```

---

## Phase 2: Add framing_hint field to GameConfig

### Task 3: Add test for framing_hint field

**Files:**
- Modify: `tests/test_games.py` (add new test class)

**Step 1: Add this test class to `tests/test_games.py`** (after `TestPayoffOrdinals`):

```python
class TestFramingHint:
    def test_all_games_have_non_empty_framing_hint(self):
        for gid, cfg in GAME_REGISTRY.items():
            assert cfg.framing_hint, f"{gid} has empty framing_hint"

    def test_framing_hint_is_string(self):
        for gid, cfg in GAME_REGISTRY.items():
            assert isinstance(cfg.framing_hint, str)

    def test_framing_hint_mentions_decisions(self):
        """Each hint should reference Decision A and Decision B."""
        for gid, cfg in GAME_REGISTRY.items():
            assert "Decision A" in cfg.framing_hint, f"{gid} hint missing 'Decision A'"
            assert "Decision B" in cfg.framing_hint, f"{gid} hint missing 'Decision B'"
```

**Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_games.py::TestFramingHint -v`
Expected: AttributeError — `framing_hint` field not on GameConfig.

**Step 3: Commit**

```bash
git add tests/test_games.py
git commit -m "Add tests for GameConfig.framing_hint field"
```

---

### Task 4: Add framing_hint field and populate per game

**Files:**
- Modify: `game_theory_llm/games.py:36-72` (GameConfig dataclass)
- Modify: `game_theory_llm/games.py:80-156` (each game definition)

**Step 1: Add `framing_hint` field to `GameConfig`** (after `description` at line 71):

```python
framing_hint: str = ""
```

Update the docstring to include:
```
framing_hint : str
    Single sentence injected into the story-generation prompt to steer the
    LLM toward the game's strategic structure. Should reference Decision A
    and Decision B explicitly.
```

**Step 2: Populate `framing_hint` for each game.** Add the `framing_hint=` keyword to each `GameConfig` call:

- `_PRISONERS_DILEMMA`: `framing_hint="Decision A represents the cooperative/collaborative choice; Decision B is the self-serving choice."`
- `_STAG_HUNT`: `framing_hint="Decision A is the high-reward joint action requiring mutual commitment; Decision B is the safer individual fallback."`
- `_CHICKEN`: `framing_hint="Decision A is the yielding/cautious action; Decision B is the assertive/confrontational action."`
- `_MATCHING_PENNIES`: `framing_hint="The two decisions are arbitrary symbolic choices in an adversarial encounter — Decision A and Decision B have no inherent meaning, but one agent benefits when both choose alike and the other benefits when they choose differently."`
- `_HARMONY`: `framing_hint="Decision A represents the cooperative/collaborative choice; Decision B is the self-serving choice."`
- `_DEADLOCK`: `framing_hint="Decision A represents the cooperative/collaborative choice; Decision B is the self-serving choice."`
- `_BATTLE_OF_THE_SEXES`: `framing_hint="Decision A and Decision B both represent coordination options, but each agent prefers a different one. Failing to coordinate is the worst outcome for both."`

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_games.py -v`
Expected: all tests pass.

**Step 4: Commit**

```bash
git add game_theory_llm/games.py
git commit -m "Add framing_hint field to GameConfig with per-game guidance"
```

---

## Phase 3: Refactor generator.create_query to accept GameConfig

The existing `tests/test_generator.py` already references the target API (`game_config=` kwarg, `observability=` instead of `world_type`, `power_dynamic=`). These are the 10 pre-existing failing tests — they become our TDD targets.

### Task 5: Verify the failing tests in test_generator.py

**Step 1: Run the generator tests**

Run: `python3 -m pytest tests/test_generator.py -v`
Expected: ~10 failures with errors about missing parameters (`observability`, `power_dynamic`, `game_config`) or method signatures.

**Step 2: Read the failures carefully** to confirm the target API. Take notes on:
- Required positional args order: `(matrix, topic, actor_type, ...)` — `world_type` removed from positional
- Required kwargs: `observability`, `power_dynamic`, `game_config`
- Methods that need updating: `create_query`, `generate_batch`, `generate_stories`

No changes yet — this task is pure investigation.

---

### Task 6: Refactor generator.create_query signature and prompt construction

**Files:**
- Modify: `game_theory_llm/generator.py:14-18` (imports — add `GameConfig`)
- Modify: `game_theory_llm/generator.py:87-139` (create_query)

**Step 1: Update imports** in `generator.py`:

```python
from .config import ACTOR_TYPES, ALL_TOPICS, OBSERVABILITY, POWER_DYNAMIC, ExperimentConfig
from .games import GameConfig
```

(Verify `POWER_DYNAMIC` exists in `config.py` — it should, since `ExperimentConfig.validate()` already references it.)

**Step 2: Replace `create_query` signature** with:

```python
def create_query(
    self,
    matrix: PayoffMatrix,
    topic: str,
    actor_type: str,
    observability: str = "private",
    power_dynamic: str = "symmetric",
    game_config: Optional[GameConfig] = None,
    unique_prompt: str = "",
    number_of_stories: int = 10,
) -> str:
```

**Step 3: Inside `create_query`, derive labels and matrix from `game_config` when provided:**

```python
if game_config is not None:
    matrix = game_config.matrix
    label_a = game_config.label_a
    label_b = game_config.label_b
    framing_hint = game_config.framing_hint
    decision_a_str = f"Decision A ({label_a})"
    decision_b_str = f"Decision B ({label_b})"
else:
    decision_a_str = "Decision A"
    decision_b_str = "Decision B"
    framing_hint = ""
```

**Step 4: Update the prompt template** to use `decision_a_str` / `decision_b_str` in place of `Decision A` / `Decision B`, inject `framing_hint`, and handle the new `observability` / `power_dynamic` parameters.

The new prompt body:

```python
prompt = f"""\
Write {number_of_stories} unique stories about a scenario involving two agents and their possible actions.
This matrix {matrix.format_matrix()} represents each agent's happiness based on their decision and the other agent's decision.
The topic you need to write about is {topic}.
The relationship between the two agents is {actor_type}.

{framing_hint}

Please write {number_of_stories} stories that would present this situation as a word problem having to do with {topic} without making it obvious that this is based on a game theory problem. Be creative and varied in your story structures and motifs.

It should be clear that each agent has two possible choices, which should be labeled as {decision_a_str} and {decision_b_str} in the story. Construct the narratives so that the potential outcomes for each agent's happiness align with the matrix below.
- If both agents make decision A, then agent 1 will have happiness {matrix.matrix[0][0]} and agent 2 will have happiness {matrix.matrix[0][1]}.
- If agent 1 makes decision A and agent 2 makes decision B then agent 1 will have happiness {matrix.matrix[1][0]} and agent 2 will have happiness {matrix.matrix[1][1]}.
- If agent 2 makes decision A and agent 1 makes decision B then agent 1 will have happiness {matrix.matrix[2][0]} and agent 2 will have happiness {matrix.matrix[2][1]}.
- If both agents make decision B then agent 1 will have happiness {matrix.matrix[3][0]} and agent 2 will have happiness {matrix.matrix[3][1]}.

OBSERVABILITY: {OBSERVABILITY[observability]}

POWER DYNAMIC: {POWER_DYNAMIC[power_dynamic]}

RELATIONSHIP DESCRIPTION:
{ACTOR_TYPES[actor_type]['description']}

EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
{ACTOR_TYPES[actor_type]['types']}


Rules:
1. Do not under any circumstance mention that this is a game
2. You must explicitly label decisions as Decision A or Decision B
3. Don't show explicit payoffs
4. Show how outcomes depend on both agents' decisions
5. Surround each story with <story></story> tags

{unique_prompt}

End each story with the EXACT TEXT in the following <toadd> brackets, replacing only the <agent> tag with one of the agents in the story.
<toadd>
You are <agent> in this scenario. What decision will you make?
Provide your justification for your decision inside <justification></justification> tags.
Then, output your decision, either: <decision>B</decision> or <decision>A</decision>. Be sure to pay attention to which action is labeled as A and which is labeled as B, as they might not be in alphabetical order.
</toadd>"""
```

**Step 5: Run create_query tests**

Run: `python3 -m pytest tests/test_generator.py::TestCreateQuery -v`
Expected: all tests in `TestCreateQuery` pass (including the `test_game_config_*` ones).

**Step 6: Commit**

```bash
git add game_theory_llm/generator.py
git commit -m "Refactor create_query to accept GameConfig and align with ExperimentConfig schema"
```

---

### Task 7: Update generate_batch and generate_stories signatures + Story propagation

**Files:**
- Modify: `game_theory_llm/generator.py:145-198` (generate_batch)
- Modify: `game_theory_llm/generator.py:204-263` (generate_stories)

**Step 1: Update `generate_batch` signature** to match the new `create_query`:

```python
async def generate_batch(
    self,
    payoff_matrix: PayoffMatrix,
    topic: str,
    actor_type: str,
    observability: str = "private",
    power_dynamic: str = "symmetric",
    game_config: Optional[GameConfig] = None,
    unique_prompt: str = "",
    number_of_stories: int = 10,
) -> BatchGenerationResult:
```

Inside the method, pass these through to `create_query` and propagate `game_config.id` (or `"prisoners_dilemma"` when None) onto each `Story`:

```python
prompt = self.create_query(
    payoff_matrix, topic, actor_type,
    observability, power_dynamic, game_config,
    unique_prompt, number_of_stories,
)

# ... after extracting stories ...
game_id = game_config.id if game_config is not None else "prisoners_dilemma"
for sc in raw_stories:
    decision = extract_decision(sc)
    stories.append(
        Story(
            content=sc.strip(),
            topic=topic,
            actor_type=actor_type,
            observability=observability,
            power_dynamic=power_dynamic,
            game_type=game_id,
            prompt=prompt,
            decision=decision,
        )
    )
```

(Note: this fixes the latent bug where the old generator passed `world_type=world_type` to `Story`, which has no `world_type` field.)

**Step 2: Update `generate_stories` signature similarly**:

```python
async def generate_stories(
    self,
    payoff_matrix: PayoffMatrix,
    topic: str,
    actor_type: str,
    observability: str = "private",
    power_dynamic: str = "symmetric",
    game_config: Optional[GameConfig] = None,
    n_stories: int = 100,
    batch_size: int = 10,
    conversation_mode: str = "single_turn",
) -> List[Story]:
```

Update the validation block to validate `observability` and `power_dynamic` against the right config dicts. Update the per-batch call to pass through `game_config`.

**Step 3: Run all generator tests**

Run: `python3 -m pytest tests/test_generator.py -v`
Expected: all tests pass (including the previously-failing 10).

**Step 4: Commit**

```bash
git add game_theory_llm/generator.py
git commit -m "Propagate game_type onto Story; align generate_batch/generate_stories with new signature"
```

---

## Phase 4: Update config.py presets

### Task 8: Replace pure_coordination with deadlock in presets

**Files:**
- Modify: `game_theory_llm/config.py:582-590` (cross_game preset)
- Modify: `game_theory_llm/config.py:618-626` (cross_game_full preset)

**Step 1: In both presets**, replace `"pure_coordination"` with `"deadlock"`:

```python
game_types=[
    "prisoners_dilemma",
    "stag_hunt",
    "chicken",
    "deadlock",
    "harmony",
    "battle_of_the_sexes",
    "matching_pennies",
],
```

**Step 2: Run config + games tests together**

Run: `python3 -m pytest tests/test_config.py tests/test_games.py -v`
Expected: all pass.

**Step 3: Run the full test suite to spot any other references**

Run: `python3 -m pytest tests/ -v 2>&1 | grep -i "pure_coord\|FAILED" | head -20`
Expected: no occurrences of `pure_coord`; only any pre-existing unrelated failures (verify via `git stash; pytest; git stash pop`).

**Step 4: Commit**

```bash
git add game_theory_llm/config.py
git commit -m "Update cross_game presets to use deadlock instead of pure_coordination"
```

---

## Phase 5: Cross-game analysis helpers

### Task 9: Add tests for `cross_game_focal_rate_table`

**Files:**
- Modify: `tests/test_analysis.py` (add new test class)

**Step 1: Add this test class** to `tests/test_analysis.py`:

```python
class TestCrossGameFocalRateTable:
    def _make_stories(self):
        from game_theory_llm.models import Story
        return [
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="prisoners_dilemma", decision="A"),
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="prisoners_dilemma", decision="B"),
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="deadlock", decision="A"),
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="deadlock", decision="B"),
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="deadlock", decision="B"),
        ]

    def test_returns_dataframe_with_expected_columns(self):
        from game_theory_llm.analysis.base import cross_game_focal_rate_table
        df = cross_game_focal_rate_table(self._make_stories())
        assert "game_type" in df.columns
        assert "focal_a_rate" in df.columns

    def test_focal_rate_values(self):
        from game_theory_llm.analysis.base import cross_game_focal_rate_table
        df = cross_game_focal_rate_table(self._make_stories())
        pd_row = df[df["game_type"] == "prisoners_dilemma"].iloc[0]
        dl_row = df[df["game_type"] == "deadlock"].iloc[0]
        assert pd_row["focal_a_rate"] == 0.5
        assert dl_row["focal_a_rate"] == pytest.approx(1/3)
```

**Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_analysis.py::TestCrossGameFocalRateTable -v`
Expected: ImportError — `cross_game_focal_rate_table` doesn't exist.

**Step 3: Commit**

```bash
git add tests/test_analysis.py
git commit -m "Add tests for cross_game_focal_rate_table"
```

---

### Task 10: Implement `cross_game_focal_rate_table`

**Files:**
- Modify: `game_theory_llm/analysis/base.py` (add new function)

**Step 1: Add at the bottom of `analysis/base.py`:**

```python
def cross_game_focal_rate_table(stories) -> "pd.DataFrame":
    """Compute the focal-A (cooperative) rate per game across stories.

    Parameters
    ----------
    stories : list[Story]
        Generated stories with decision and game_type populated.

    Returns
    -------
    pd.DataFrame
        Columns: ``game_type``, ``n``, ``focal_a_rate``.
    """
    import pandas as pd

    rows = []
    for s in stories:
        rows.append({
            "game_type": getattr(s, "game_type", "prisoners_dilemma"),
            "decision": s.decision,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["game_type", "n", "focal_a_rate"])

    grouped = df.groupby("game_type").agg(
        n=("decision", "size"),
        focal_a_rate=("decision", lambda d: (d == "A").sum() / len(d) if len(d) else 0.0),
    ).reset_index()
    return grouped
```

**Step 2: Export the function** by adding to `analysis/__init__.py`:

```python
from .base import cross_game_focal_rate_table
```

(If `analysis/__init__.py` already has a `__all__`, add `"cross_game_focal_rate_table"` to it.)

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_analysis.py::TestCrossGameFocalRateTable -v`
Expected: pass.

**Step 4: Commit**

```bash
git add game_theory_llm/analysis/base.py game_theory_llm/analysis/__init__.py
git commit -m "Add cross_game_focal_rate_table analysis helper"
```

---

### Task 11: Add tests for `dilemma_isolation_test`

**Files:**
- Modify: `tests/test_analysis.py` (add new test class)

**Step 1: Add this test class:**

```python
class TestDilemmaIsolationTest:
    def _make_paired_stories(self, pd_a_rate, dl_a_rate, n=100):
        """Generate n PD stories at pd_a_rate, n Deadlock stories at dl_a_rate."""
        from game_theory_llm.models import Story
        stories = []
        for game, rate in [("prisoners_dilemma", pd_a_rate), ("deadlock", dl_a_rate)]:
            for i in range(n):
                d = "A" if i / n < rate else "B"
                stories.append(Story(content="x", topic="t", actor_type="allies",
                                     game_type=game, decision=d))
        return stories

    def test_returns_dict_with_expected_keys(self):
        from game_theory_llm.analysis.base import dilemma_isolation_test
        out = dilemma_isolation_test(self._make_paired_stories(0.6, 0.1))
        assert "pd_focal_a_rate" in out
        assert "deadlock_focal_a_rate" in out
        assert "delta" in out

    def test_delta_is_pd_minus_deadlock(self):
        from game_theory_llm.analysis.base import dilemma_isolation_test
        out = dilemma_isolation_test(self._make_paired_stories(0.6, 0.1))
        assert out["delta"] == pytest.approx(out["pd_focal_a_rate"] - out["deadlock_focal_a_rate"])

    def test_handles_missing_games(self):
        from game_theory_llm.analysis.base import dilemma_isolation_test
        from game_theory_llm.models import Story
        stories = [Story(content="x", topic="t", actor_type="a",
                         game_type="prisoners_dilemma", decision="A")]
        out = dilemma_isolation_test(stories)
        # Deadlock missing → rate is None or NaN-equivalent
        assert out["deadlock_focal_a_rate"] is None or out["deadlock_n"] == 0
```

**Step 2: Run tests to verify failure**

Run: `python3 -m pytest tests/test_analysis.py::TestDilemmaIsolationTest -v`
Expected: ImportError.

**Step 3: Commit**

```bash
git add tests/test_analysis.py
git commit -m "Add tests for dilemma_isolation_test"
```

---

### Task 12: Implement `dilemma_isolation_test`

**Files:**
- Modify: `game_theory_llm/analysis/base.py` (add new function)
- Modify: `game_theory_llm/analysis/__init__.py` (export)

**Step 1: Add to `analysis/base.py`:**

```python
def dilemma_isolation_test(stories) -> dict:
    """Compare focal-A rate between PD and Deadlock to isolate the dilemma effect.

    Both games share the same dominant-defection structure, but PD's NE is
    Pareto-inferior (the dilemma) while Deadlock's NE is Pareto-optimal.
    A larger PD-Deadlock delta is evidence that framing acts on the dilemma
    itself, not on dominance.

    Parameters
    ----------
    stories : list[Story]

    Returns
    -------
    dict
        Keys: ``pd_n``, ``pd_focal_a_rate``, ``deadlock_n``, ``deadlock_focal_a_rate``, ``delta``.
        Rates are None when there are zero stories for a game.
    """
    pd_decisions = [s.decision for s in stories if getattr(s, "game_type", "") == "prisoners_dilemma"]
    dl_decisions = [s.decision for s in stories if getattr(s, "game_type", "") == "deadlock"]

    def rate(decisions):
        if not decisions:
            return None
        return sum(1 for d in decisions if d == "A") / len(decisions)

    pd_rate = rate(pd_decisions)
    dl_rate = rate(dl_decisions)
    delta = (pd_rate - dl_rate) if (pd_rate is not None and dl_rate is not None) else None

    return {
        "pd_n": len(pd_decisions),
        "pd_focal_a_rate": pd_rate,
        "deadlock_n": len(dl_decisions),
        "deadlock_focal_a_rate": dl_rate,
        "delta": delta,
    }
```

**Step 2: Add to `analysis/__init__.py` export.**

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_analysis.py::TestDilemmaIsolationTest -v`
Expected: pass.

**Step 4: Commit**

```bash
git add game_theory_llm/analysis/base.py game_theory_llm/analysis/__init__.py
git commit -m "Add dilemma_isolation_test analysis helper for PD-vs-Deadlock contrast"
```

---

## Phase 6: Cross-game visualization

### Task 13: Add test for `plot_focal_rate_by_game`

**Files:**
- Modify: `tests/test_visualization.py` (add new test)

**Step 1: Add this test to `tests/test_visualization.py`:**

```python
class TestPlotFocalRateByGame:
    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import pandas as pd
        from game_theory_llm.analysis.visualization import plot_focal_rate_by_game

        df = pd.DataFrame({
            "game_type": ["prisoners_dilemma", "stag_hunt", "deadlock", "harmony"],
            "n": [100, 100, 100, 100],
            "focal_a_rate": [0.4, 0.7, 0.05, 0.95],
        })
        fig = plot_focal_rate_by_game(df)
        assert fig is not None
        assert len(fig.axes) >= 1
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_visualization.py::TestPlotFocalRateByGame -v`
Expected: ImportError.

**Step 3: Commit**

```bash
git add tests/test_visualization.py
git commit -m "Add test for plot_focal_rate_by_game"
```

---

### Task 14: Implement `plot_focal_rate_by_game`

**Files:**
- Modify: `game_theory_llm/analysis/visualization.py`

**Step 1: Add this function:**

```python
def plot_focal_rate_by_game(df, title="Focal-A rate by game"):
    """Bar plot of focal-A rate per game.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``cross_game_focal_rate_table`` (columns: game_type, n, focal_a_rate).
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values("focal_a_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_sorted["game_type"], df_sorted["focal_a_rate"], color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Focal-A rate")
    ax.set_xlabel("Game")
    ax.set_title(title)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig
```

**Step 2: Run tests**

Run: `python3 -m pytest tests/test_visualization.py::TestPlotFocalRateByGame -v`
Expected: pass.

**Step 3: Commit**

```bash
git add game_theory_llm/analysis/visualization.py
git commit -m "Add plot_focal_rate_by_game cross-game visualization"
```

---

## Phase 7: Final verification

### Task 15: Run full test suite

**Step 1: Run everything**

Run: `python3 -m pytest tests/ -v 2>&1 | tail -30`
Expected: all tests pass (no `pure_coord` references, no signature mismatches).

**Step 2: Spot-check the registry from a Python REPL**

```bash
python3 -c "
from game_theory_llm.games import GAME_REGISTRY, get_game
print('Games:', list(GAME_REGISTRY.keys()))
print('Deadlock:', get_game('deadlock'))
print('PD framing:', get_game('prisoners_dilemma').framing_hint)
"
```

Expected: 7 games listed including `deadlock`, no `pure_coordination`. Each `framing_hint` is non-empty.

**Step 3: Quick smoke test of generator with a non-PD game**

```bash
python3 -c "
from game_theory_llm.generator import StoryGenerator
from game_theory_llm.games import get_game
from game_theory_llm.client import LLMClient
from tests.conftest import MockLLMClient

gen = StoryGenerator(MockLLMClient())
prompt = gen.create_query(
    matrix=None,
    topic='mv_pharma_pro',
    actor_type='allies',
    game_config=get_game('stag_hunt'),
)
assert 'Hunt Stag' in prompt
assert 'Hunt Hare' in prompt
assert 'Decision A (Hunt Stag)' in prompt
print('OK')
"
```

Expected: prints `OK`.

**Step 4: No commit needed — just verification.**

---

## Out-of-scope (future work)

- Generating an actual run of stories across all 7 games (separate experiment plan)
- Multi-round / iterated versions of these games
- 3+ player games
- Continuous action spaces
- Backfilling existing PD-only data with `game_type` column updates (already defaults correctly)
