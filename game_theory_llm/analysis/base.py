# game_theory_llm/analysis/base.py
"""Core story analysis — process stories, compute proportions, summaries.

The paper presents every vignette to each LLM *twice*:
  1. Original:  Decision A = Cooperate, Decision B = Defect
  2. Swapped:   Decision B = Cooperate, Decision A = Defect

``build_dataframe`` runs both passes and returns a single DataFrame with
``decision_{model}`` (original) and ``decision_swapped_{model}`` (swapped)
columns, plus matching ``response_*`` columns.  This DataFrame is the
primary input for every analysis and figure in the paper.

Cooperation convention:
  - Original:  decision == 'A'  →  Cooperate
  - Swapped:   decision_swapped == 'B'  →  Cooperate
"""

from typing import Callable, Dict, List, Optional

import pandas as pd

from .._logging import get_logger
from ..client import LLMClient
from ..decision_parser import extract_decision
from ..models import AnalysisResult, PayoffMatrix, Story

logger = get_logger(__name__)

# Game IDs used by the dilemma isolation analysis. Must match GAME_REGISTRY keys.
_PD_GAME_ID = "prisoners_dilemma"
_DEADLOCK_GAME_ID = "deadlock"

_MODELS = ("llama", "claude", "gpt4")

# Categorical columns that appear in every analysis DataFrame.
_CATEGORIES = ("topic", "actor_type", "observability", "power_dynamic", "game_type", "conversation_mode")


def _swap_labels(text: str) -> str:
    """Swap every occurrence of 'Decision A' and 'Decision B' in *text*.

    Uses a neutral placeholder to avoid double-replacement.
    """
    text = text.replace("Decision A", "Decision __SWAP__")
    text = text.replace("Decision B", "Decision A")
    text = text.replace("Decision __SWAP__", "Decision B")
    return text


class StoryAnalyzer:
    """Processes stories through LLMs and computes decision distributions.

    Parameters
    ----------
    client : LLMClient
        The client used to send prompts to LLMs.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    # ------------------------------------------------------------------
    # Single-story processing
    # ------------------------------------------------------------------

    async def process_story(self, story: Story) -> Dict[str, Optional[str]]:
        """Send *story* to every configured LLM and extract decisions.

        The story content already contains a character-specific elicitation
        block at the end (added by the generator). We send it as-is so the
        test subject sees the question framed within the scenario.

        Returns a dict with keys ``"llama"``, ``"claude"``, ``"gpt4"``
        (decisions) and ``"llama_response"`` etc. (raw text).
        """
        logger.debug("Processing story (ID: %s)", id(story))
        prompt = story.content
        try:
            responses = await self.client.generate(prompt)
            decisions: Dict[str, Optional[str]] = {}
            for model, response in responses.items():
                decisions[model] = extract_decision(response) if response else None
                decisions[f"{model}_response"] = response
            logger.info(
                "Extracted decisions: %s",
                {k: v for k, v in decisions.items() if "_response" not in k},
            )
            return decisions
        except Exception as e:
            logger.error("Error processing story %s: %s", id(story), e, exc_info=True)
            return {m: None for m in self.client.models}

    async def process_story_swapped(self, story: Story) -> Dict[str, Optional[str]]:
        """Like :meth:`process_story` but with Decision A/B labels swapped.

        Returns keys ``"llama"``, ``"claude"``, ``"gpt4"`` (raw A/B from the
        swapped presentation) and corresponding ``"*_response"`` keys.

        To normalise to Cooperate/Defect: ``decision == 'B'`` in the swapped
        result corresponds to Cooperate (since B is now the cooperative option).
        """
        logger.debug("Processing swapped story (ID: %s)", id(story))
        prompt = _swap_labels(story.content)
        try:
            responses = await self.client.generate(prompt)
            decisions: Dict[str, Optional[str]] = {}
            for model, response in responses.items():
                decisions[model] = extract_decision(response) if response else None
                decisions[f"{model}_response"] = response
            return decisions
        except Exception as e:
            logger.error(
                "Error processing swapped story %s: %s", id(story), e, exc_info=True
            )
            return {m: None for m in self.client.models}

    # ------------------------------------------------------------------
    # Full DataFrame construction (primary analysis method)
    # ------------------------------------------------------------------

    async def build_dataframe(
        self,
        stories: List[Story],
        include_swapped: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> pd.DataFrame:
        """Run both original and swapped analysis, return a single DataFrame.

        The returned DataFrame has one row per story with columns:
        ``topic``, ``actor_type``, ``observability``, ``power_dynamic``,
        ``story_content``, ``decision_{model}``, ``response_{model}``,
        and (when *include_swapped*) ``decision_swapped_{model}``,
        ``response_swapped_{model}``.

        Cooperation convention
        ---------------------
        - Original: ``decision_{model} == 'A'``  → Cooperate
        - Swapped:  ``decision_swapped_{model} == 'B'``  → Cooperate
        """
        logger.info(
            "Building analysis dataframe for %d stories (swapped=%s)",
            len(stories), include_swapped,
        )
        rows: List[dict] = []
        for i, story in enumerate(stories, 1):
            logger.debug("Processing story %d/%d", i, len(stories))
            orig = await self.process_story(story)
            row: dict = {
                "topic": story.topic,
                "actor_type": story.actor_type,
                "observability": story.observability,
                "power_dynamic": story.power_dynamic,
                "game_type": getattr(story, "game_type", "prisoners_dilemma"),
                "conversation_mode": getattr(story, "conversation_mode", "single_turn"),
                "story_content": story.content,
            }
            for m in _MODELS:
                row[f"decision_{m}"] = orig.get(m)
                row[f"response_{m}"] = orig.get(f"{m}_response")

            if include_swapped:
                swapped = await self.process_story_swapped(story)
                for m in _MODELS:
                    row[f"decision_swapped_{m}"] = swapped.get(m)
                    row[f"response_swapped_{m}"] = swapped.get(f"{m}_response")

            rows.append(row)
            if progress_callback:
                progress_callback(1)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Legacy batch analysis (returns AnalysisResult)
    # ------------------------------------------------------------------

    async def analyze_stories(
        self,
        stories: List[Story],
        payoff_matrix: Optional[PayoffMatrix] = None,
        df_existing: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> AnalysisResult:
        """Analyse a collection of stories and return multi-model results.

        For full paper reproduction (including swapped analysis) use
        :meth:`build_dataframe` instead.
        """
        logger.info("Starting analysis of %d stories", len(stories))

        if df_existing is not None:
            logger.info("Using existing dataframe")
            df = df_existing
            decisions = {
                model: list(df[f"decision_{model}"])
                for model in _MODELS
            }
        else:
            raw: Dict[str, list] = {
                k: []
                for k in [
                    *_MODELS,
                    *[f"{m}_response" for m in _MODELS],
                ]
            }
            for i, story in enumerate(stories, 1):
                logger.debug("Processing story %d/%d", i, len(stories))
                result = await self.process_story(story)
                for key, value in result.items():
                    raw[key].append(value)
                if progress_callback:
                    progress_callback(1)

            df = pd.DataFrame({
                "topic": [s.topic for s in stories],
                "actor_type": [s.actor_type for s in stories],
                "observability": [s.observability for s in stories],
                "power_dynamic": [s.power_dynamic for s in stories],
                "game_type": [getattr(s, "game_type", "prisoners_dilemma") for s in stories],
                "conversation_mode": [getattr(s, "conversation_mode", "single_turn") for s in stories],
            })
            for key, values in raw.items():
                col = f"response_{key}" if "response" in key else f"decision_{key}"
                df[col] = values
            decisions = {m: list(df[f"decision_{m}"]) for m in _MODELS}

        proportions: Dict[str, dict] = {}
        by_topic: Dict[str, dict] = {}
        by_observability: Dict[str, dict] = {}
        by_power: Dict[str, dict] = {}
        by_actor: Dict[str, dict] = {}
        by_game: Dict[str, dict] = {}
        summaries: Dict[str, List[str]] = {}

        for model in _MODELS:
            col = f"decision_{model}"
            proportions[model] = df[col].value_counts(normalize=True).to_dict()

            by_topic[model] = {
                tp: df.loc[df["topic"] == tp, col].value_counts(normalize=True).to_dict()
                for tp in df["topic"].unique()
            }
            by_observability[model] = {
                o: df.loc[df["observability"] == o, col].value_counts(normalize=True).to_dict()
                for o in df["observability"].unique()
            }
            by_power[model] = {
                p: df.loc[df["power_dynamic"] == p, col].value_counts(normalize=True).to_dict()
                for p in df["power_dynamic"].unique()
            }
            by_actor[model] = {
                a: df.loc[df["actor_type"] == a, col].value_counts(normalize=True).to_dict()
                for a in df["actor_type"].unique()
            }
            if "game_type" in df.columns:
                by_game[model] = {
                    g: df.loc[df["game_type"] == g, col].value_counts(normalize=True).to_dict()
                    for g in df["game_type"].unique()
                }
            summaries[model] = self.generate_summaries(df, payoff_matrix, model)

        logger.info("Analysis complete")
        return AnalysisResult(
            stories=stories,
            decisions=decisions,
            summaries=summaries,
            proportions=proportions,
            by_topic=by_topic,
            by_observability=by_observability,
            by_power=by_power,
            by_actor=by_actor,
            by_game=by_game,
        )

    # ------------------------------------------------------------------
    # Summaries & utilities
    # ------------------------------------------------------------------

    def generate_summaries(
        self,
        df: pd.DataFrame,
        payoff_matrix: Optional[PayoffMatrix],
        model: str,
    ) -> List[str]:
        """Return textual summaries for one model's decisions."""
        col = f"decision_{model}"
        sums: List[str] = []
        dist = df[col].value_counts(normalize=True).to_dict()
        sums.append(f"Overall Decision Distribution: {dist}")

        for cat in _CATEGORIES:
            if cat in df.columns:
                for val in df[cat].unique():
                    sub = df.loc[df[cat] == val, col].value_counts(normalize=True).to_dict()
                    sums.append(f"Distribution by {cat}={val}: {sub}")

        if payoff_matrix:
            prob_a = df[col].eq("A").mean()
            sums.append(f"Probability of choosing A: {prob_a:.2f}")
            eu = self.calculate_expected_utility(prob_a, payoff_matrix)
            sums.append(f"Expected utility: {eu:.2f}")

        return sums

    @staticmethod
    def calculate_expected_utility(prob_a: float, matrix: PayoffMatrix) -> float:
        """Expected utility given Pr(A) in a symmetric 2x2 game."""
        prob_b = 1 - prob_a
        return (
            prob_a * prob_a * matrix.matrix[0][0]
            + prob_a * prob_b * matrix.matrix[1][0]
            + prob_b * prob_a * matrix.matrix[2][0]
            + prob_b * prob_b * matrix.matrix[3][0]
        )


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
        n=("decision", lambda d: d.dropna().size),
        focal_a_rate=("decision", lambda d: (d.dropna() == "A").mean() if d.dropna().size else 0.0),
    ).reset_index()
    return grouped


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
    pd_decisions = [s.decision for s in stories if getattr(s, "game_type", "") == _PD_GAME_ID]
    dl_decisions = [s.decision for s in stories if getattr(s, "game_type", "") == _DEADLOCK_GAME_ID]

    def rate(decisions):
        valid = [d for d in decisions if d is not None]
        if not valid:
            return None
        return sum(1 for d in valid if d == "A") / len(valid)

    pd_rate = rate(pd_decisions)
    dl_rate = rate(dl_decisions)
    delta = (pd_rate - dl_rate) if (pd_rate is not None and dl_rate is not None) else None

    return {
        "pd_n": sum(1 for d in pd_decisions if d is not None),
        "pd_focal_a_rate": pd_rate,
        "deadlock_n": sum(1 for d in dl_decisions if d is not None),
        "deadlock_focal_a_rate": dl_rate,
        "delta": delta,
    }
