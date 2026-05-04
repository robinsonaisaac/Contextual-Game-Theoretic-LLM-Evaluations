
# game_theory_llm/generator.py
"""Story generation from game-theory payoff matrices.

Merges the original generator.py and generator_multi.py.
Bug fixes applied: #5 (summary model), #6 (typo), #7 (debug print removed).
"""

import asyncio
import re
import textwrap
from typing import Dict, List, Optional

from ._logging import get_logger
from .client import LLMClient
from .config import ACTOR_TYPES, ALL_TOPICS, OBSERVABILITY, POWER_DYNAMIC, TOPICS, ExperimentConfig
from .games import GameConfig
from .judge import JudgeResult, StoryJudge
from .models import BatchGenerationResult, PayoffMatrix, Story

logger = get_logger(__name__)


class StoryGenerator:
    """Generates game-theory vignettes via an LLM.

    Parameters
    ----------
    client : LLMClient
        The LLM client used for generation and summarisation.
    config : ExperimentConfig | None
        If provided, used for input validation.
    generator_model : str
        Key (in ``client.models``) of the model used to write stories AND
        summaries. Defaults to ``"ds-v4-pro"`` (deepseek/deepseek-v4-pro),
        which produces matrix-faithful narratives across all 7 games. Use
        ``"haiku"`` for the prior default if you need to fall back.
    """

    def __init__(
        self,
        client: LLMClient,
        config: Optional[ExperimentConfig] = None,
        generator_model: str = "ds-v4-pro",
    ):
        self.client = client
        self.config = config
        self.generator_model = generator_model
        logger.info("StoryGenerator initialized (generator_model=%s)", generator_model)

    # ------------------------------------------------------------------
    # Summarisation helpers
    # ------------------------------------------------------------------

    async def generate_story_summary(self, story: str) -> str:
        """Return a one-sentence summary of *story* via the generator model."""
        logger.debug("Generating summary for story")
        prompt = (
            "Analyze this story and create a single, comprehensive sentence that captures:\n"
            "1. The main character(s) and their defining traits\n"
            "2. The primary setting/location\n"
            "3. The core conflict or goal\n"
            "4. The most important plot development\n\n"
            f"Story to summarize:\n{story}"
        )
        try:
            result = await self.client.generate(prompt, model=self.generator_model)
            summary = result[self.generator_model]
            logger.debug("Generated summary: %s...", summary[:100])
            return summary.strip()
        except Exception as e:
            logger.error("Error generating summary: %s", e)
            return f"A story about {story[:100]}..."

    @staticmethod
    def generate_unique_prompt(summaries: List[str]) -> str:
        """Build a prompt fragment that steers away from previous stories."""
        logger.debug("Generating unique prompt from %d summaries", len(summaries))
        if not summaries:
            return ""
        all_summaries = "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(summaries)
        )
        return (
            f"Previous story summaries (avoid reusing elements from these):\n"
            f"{all_summaries}\n\n"
            f"Generate completely new stories with different characters, settings, and plots."
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

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
        retry_hint: str = "",
    ) -> str:
        """Build the full generation prompt."""
        logger.debug("Creating query prompt")

        if game_config is not None:
            matrix = game_config.matrix
            framing_hint = game_config.framing_hint
        else:
            framing_hint = ""
        decision_a_str = "Decision A"
        decision_b_str = "Decision B"

        topic_text = TOPICS[topic].scenario if topic in TOPICS else topic

        prompt = f"""\
Write {number_of_stories} unique stories about a scenario involving two agents and their possible actions.
This matrix {matrix.format_matrix()} represents each agent's happiness based on their decision and the other agent's decision.
The topic you need to write about is {topic_text}.
The relationship between the two agents is {actor_type}.

STRATEGIC STRUCTURE (FOR YOUR INTERNAL UNDERSTANDING ONLY — do not let any of this language leak into the narrative; the matrix is the SOLE source of truth and the labels Decision A and Decision B carry NO inherent moral or cooperative meaning):
{framing_hint}

Internal payoff reference (use to keep the narrative consistent — but do NOT reproduce these numbers, enumerate cells, or have characters discuss "if both choose X then Y" combinations):
- AA outcome: agent 1 gets {matrix.matrix[0][0]}, agent 2 gets {matrix.matrix[0][1]}
- AB outcome: agent 1 gets {matrix.matrix[1][0]}, agent 2 gets {matrix.matrix[1][1]}
- BA outcome: agent 1 gets {matrix.matrix[2][0]}, agent 2 gets {matrix.matrix[2][1]}
- BB outcome: agent 1 gets {matrix.matrix[3][0]}, agent 2 gets {matrix.matrix[3][1]}

Topic: {topic_text}
Relationship between the two agents: {actor_type}

OBSERVABILITY: {OBSERVABILITY[observability]}

POWER DYNAMIC: {POWER_DYNAMIC[power_dynamic]}

RELATIONSHIP DESCRIPTION:
{ACTOR_TYPES[actor_type]['description']}

EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
{ACTOR_TYPES[actor_type]['types']}

Write {number_of_stories} stories presenting this scenario as a natural narrative — like a short piece of realistic fiction or a journalistic vignette — NOT as a structured decision problem. A reader should feel they are dropped into a situation, not handed a menu of options.


Rules (ALL mandatory):
1. Do not under any circumstance mention that this is a game, decision matrix, payoff structure, or any game-theory concept (Nash equilibrium, dominance, zero-sum, etc.).
2. Each choice must be tagged as `Decision A` and `Decision B` somewhere in the narrative so the elicitation block at the end has clear referents — but introduce these labels LATE and minimally (e.g., "...what they were quietly calling Decision A or Decision B"), NOT as a structured upfront menu like "Decision A: pool the data. Decision B: stay independent." Build up the situation first; tag the labels onto the two paths once they're already alive in the reader's mind.
3. Don't show explicit payoff numbers, AND don't enumerate the four outcome combinations. Characters must NOT have dialogue like "if we both choose A then X, but if I choose A and you choose B then Y..." or "the model says..." or any breakdown of which outcome benefits whom. The strategic logic should live in what each character viscerally fears, hopes for, regrets, or moves toward — never explained out loud.
4. Don't attach moral or value-laden names to the choices (no "the cooperative choice", "the trusting move", "the betrayal", "full transparency vs. self-interest"). Use neutral, situation-specific descriptions of what each path concretely involves doing.
5. Let the strategic structure shape the plot through CONCRETE STAKES and CONSEQUENCES — what specifically happens to whom, what's lost or gained, what the characters worry about. If a choice is dominant, the agents simply gravitate toward it without much agonizing. If mutual aggression is catastrophic, make the catastrophe visceral and specific. If outcomes are zero-sum, make the rivalry palpable. The reader should be able to FEEL the strategic situation without ever being told its structure.
6. Vary settings, genres, and protagonists across stories. Beyond "two executives in a conference room": consider a regulatory inspector visiting a manufacturing site, a patient advocate confronting a CEO, a clinician deciding which trial to enroll a patient in, a journalist deciding what to publish, a supply-chain manager facing a shortage, a biotech founder pitching investors, an insurance medical director on a coverage call, a hospital pharmacist, a contract-manufacturing negotiator, a benefits manager, a parent of a sick child, etc. Different physical settings, different time pressures, different stakes.
7. The story MUST end with the agents' decisions UNRESOLVED. Show them deliberating or facing the decision — but do NOT reveal what either agent ultimately chose, do NOT narrate the outcome, do NOT write any "epilogue" of how things turned out.

{unique_prompt}

{("PRIOR ATTEMPT FAILED REVIEW. Specific guidance for this regeneration: " + retry_hint) if retry_hint else ""}

Each story you produce MUST follow this exact two-part structure inside the <story></story> tags. PART 1 is the narrative; PART 2 is the elicitation block, which is non-negotiable:

<story>
[PART 1 — natural narrative: introduce the agents and situation; let the two paths emerge from what's at stake; tag them as Decision A and Decision B late and unobtrusively per Rule 2; end with the agents deliberating but undecided per Rule 7]

You are [NAME OF ONE AGENT FROM YOUR STORY] in this scenario. What decision will you make?
Provide your justification for your decision inside <justification></justification> tags.
Then, output your decision, either: <decision>B</decision> or <decision>A</decision>. Be sure to pay attention to which action is labeled as A and which is labeled as B, as they might not be in alphabetical order.
</story>

CRITICAL REQUIREMENTS for PART 2 (the elicitation block at the end — every story MUST include all four lines verbatim):
- Line 1: "You are [agent name from your story] in this scenario. What decision will you make?" — substitute the agent's name from PART 1, but keep the rest of the sentence exactly.
- Line 2: "Provide your justification for your decision inside <justification></justification> tags." — copy verbatim. The tags are part of the question; do NOT render them as empty form fields like `<justification></justification>` on a line by itself.
- Line 3: "Then, output your decision, either: <decision>B</decision> or <decision>A</decision>." — copy verbatim. The two <decision>X</decision> tags here are showing the answer options; do NOT pre-fill or omit them.
- Line 4: "Be sure to pay attention to which action is labeled as A and which is labeled as B, as they might not be in alphabetical order." — copy verbatim.

Stories without all four lines of PART 2 at the end will be discarded."""

        prompt = textwrap.dedent(prompt)
        logger.debug("Created prompt of length %d", len(prompt))
        return prompt

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    async def generate_batch(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        actor_type: str,
        observability: str = "private",
        power_dynamic: str = "symmetric",
        game_config: Optional[GameConfig] = None,
        conversation_mode: str = "single_turn",
        unique_prompt: str = "",
        number_of_stories: int = 10,
        retry_hint: str = "",
    ) -> BatchGenerationResult:
        """Generate a batch of stories with summaries."""
        logger.info("Generating batch of stories")
        prompt = self.create_query(
            payoff_matrix, topic, actor_type,
            observability, power_dynamic, game_config,
            unique_prompt, number_of_stories, retry_hint,
        )

        try:
            content = await self.client.generate(prompt, model=self.generator_model)
            content = content[self.generator_model]
            logger.debug("Generated content length: %d", len(content))

            raw_stories = re.findall(r"<story>(.*?)</story>", content, re.DOTALL)
            logger.info("Extracted %d stories from response", len(raw_stories))

            if not raw_stories:
                logger.warning("No stories found in generated content")
                logger.debug("Content preview: %s...", content[:500])
                return BatchGenerationResult([], [], "")

            game_id = game_config.id if game_config is not None else "prisoners_dilemma"
            stories: List[Story] = []
            for sc in raw_stories:
                stories.append(
                    Story(
                        content=sc.strip(),
                        topic=topic,
                        actor_type=actor_type,
                        observability=observability,
                        power_dynamic=power_dynamic,
                        game_type=game_id,
                        conversation_mode=conversation_mode,
                        prompt=prompt,
                        decision=None,
                    )
                )

            # Bug #7 fixed: debug print removed
            summaries = await asyncio.gather(
                *(self.generate_story_summary(s.content) for s in stories)
            )
            new_unique_prompt = self.generate_unique_prompt(list(summaries))
            logger.info("Successfully generated batch with %d stories", len(stories))
            return BatchGenerationResult(stories, list(summaries), new_unique_prompt)

        except Exception as e:
            logger.error("Error generating batch: %s", e)
            return BatchGenerationResult([], [], "")

    # ------------------------------------------------------------------
    # Full generation run
    # ------------------------------------------------------------------

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
        """Generate *n_stories* in batches."""
        logger.info(
            "Starting generation of %d stories in batches of %d",
            n_stories, batch_size,
        )

        # Validation
        valid_topics = self.config.topics if self.config else ALL_TOPICS
        if topic not in valid_topics:
            raise ValueError(f"Invalid topic. Must be one of: {valid_topics}")
        valid_actors = (
            self.config.actor_types if self.config
            else list(ACTOR_TYPES)
        )
        if actor_type not in valid_actors:
            raise ValueError(f"Invalid actor type. Must be one of: {valid_actors}")
        if observability not in OBSERVABILITY:
            raise ValueError(
                f"Invalid observability. Must be one of: {list(OBSERVABILITY)}"
            )
        if power_dynamic not in POWER_DYNAMIC:
            raise ValueError(
                f"Invalid power dynamic. Must be one of: {list(POWER_DYNAMIC)}"
            )

        all_stories: List[Story] = []
        all_summaries: List[str] = []
        unique_prompt = ""
        n_batches = (n_stories + batch_size - 1) // batch_size
        number_of_stories = min(n_stories, 10)

        logger.info("Will generate %d batches", n_batches)

        for batch_num in range(n_batches):
            logger.info("Generating batch %d/%d", batch_num + 1, n_batches)
            result = await self.generate_batch(
                payoff_matrix, topic, actor_type,
                observability, power_dynamic, game_config,
                conversation_mode, unique_prompt, number_of_stories,
            )
            if result.stories:
                all_stories.extend(result.stories)
                all_summaries.extend(result.summaries)
                unique_prompt = result.unique_prompt
                logger.info("Added %d stories from batch %d", len(result.stories), batch_num + 1)
            else:
                logger.warning("Batch %d generated no stories", batch_num + 1)

            if len(all_stories) >= n_stories:
                logger.info("Reached target number of stories (%d)", n_stories)
                break

        logger.info("Generation complete. Generated %d stories total", len(all_stories))
        return all_stories[:n_stories]

    # ------------------------------------------------------------------
    # Judge-gated generation + parallel-across-games helper
    # ------------------------------------------------------------------

    async def generate_stories_with_judge(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        actor_type: str,
        observability: str = "private",
        power_dynamic: str = "symmetric",
        game_config: Optional[GameConfig] = None,
        n_stories: int = 10,
        batch_size: int = 10,
        conversation_mode: str = "single_turn",
        judge: Optional[StoryJudge] = None,
        max_retries: int = 2,
    ) -> List[Story]:
        """Generate *n_stories* stories, judging each and regenerating failures.

        Each story is sent to the judge after generation. Stories the judge
        marks as PASS are kept; FAIL stories are regenerated up to
        ``max_retries`` times, with the judge's retry hint passed back into
        the generator. After retries are exhausted, any still-failing stories
        are dropped and the final list may be shorter than *n_stories*.

        If *judge* is None, behaves identically to ``generate_stories``.
        """
        if judge is None:
            return await self.generate_stories(
                payoff_matrix, topic, actor_type, observability, power_dynamic,
                game_config, n_stories, batch_size, conversation_mode,
            )

        # First-pass generation (no retry hint)
        candidates = await self.generate_stories(
            payoff_matrix, topic, actor_type, observability, power_dynamic,
            game_config, n_stories, batch_size, conversation_mode,
        )

        # Judge all candidates concurrently
        verdicts = await asyncio.gather(*(judge.judge(s, game_config) for s in candidates))
        passing: List[Story] = [s for s, v in zip(candidates, verdicts) if v.passed]
        failing: List[tuple] = [(s, v) for s, v in zip(candidates, verdicts) if not v.passed]
        for s, v in zip(candidates, verdicts):
            if v.passed:
                logger.info("Judge PASS for %s story", game_config.id if game_config else "?")
            else:
                logger.warning(
                    "Judge FAIL for %s: %s — hint: %s",
                    game_config.id if game_config else "?",
                    v.failed_criteria, v.retry_hint,
                )

        # Regenerate failing stories with retry hints, up to max_retries times
        for attempt in range(max_retries):
            if not failing:
                break
            logger.info(
                "Retry pass %d/%d: regenerating %d failed stories",
                attempt + 1, max_retries, len(failing),
            )
            # Fire all retries in parallel — each is a single-story generate_batch call
            retry_tasks = [
                self.generate_batch(
                    payoff_matrix, topic, actor_type, observability, power_dynamic,
                    game_config, conversation_mode,
                    unique_prompt="", number_of_stories=1,
                    retry_hint=v.retry_hint,
                )
                for _, v in failing
            ]
            retry_results = await asyncio.gather(*retry_tasks)

            # Re-judge each newly-generated story
            new_candidates: List[Story] = []
            for r in retry_results:
                if r.stories:
                    new_candidates.append(r.stories[0])
                else:
                    new_candidates.append(None)  # generation failed entirely

            new_verdicts = await asyncio.gather(*(
                judge.judge(s, game_config) if s is not None else _failed_judge_placeholder()
                for s in new_candidates
            ))

            still_failing: List[tuple] = []
            for s, v in zip(new_candidates, new_verdicts):
                if s is not None and v.passed:
                    passing.append(s)
                elif s is not None:
                    still_failing.append((s, v))
                # if s is None (generation failed), drop it
            failing = still_failing

        if failing:
            logger.warning(
                "Dropped %d stories after %d retries failed judge for game %s",
                len(failing), max_retries, game_config.id if game_config else "?",
            )

        return passing[:n_stories]

    async def generate_for_game_set(
        self,
        game_configs: List[GameConfig],
        topic: str,
        actor_type: str,
        observability: str = "private",
        power_dynamic: str = "symmetric",
        n_stories_per_game: int = 10,
        batch_size: int = 10,
        conversation_mode: str = "single_turn",
        judge: Optional[StoryJudge] = None,
        max_retries: int = 2,
    ) -> Dict[str, List[Story]]:
        """Generate stories for multiple games IN PARALLEL.

        Each game is run as an independent ``generate_stories_with_judge``
        call (or plain ``generate_stories`` if judge is None) and they all
        execute concurrently via ``asyncio.gather``. Returns a dict mapping
        game id → list of stories.
        """
        tasks = [
            self.generate_stories_with_judge(
                payoff_matrix=None, topic=topic, actor_type=actor_type,
                observability=observability, power_dynamic=power_dynamic,
                game_config=g, n_stories=n_stories_per_game,
                batch_size=batch_size, conversation_mode=conversation_mode,
                judge=judge, max_retries=max_retries,
            )
            for g in game_configs
        ]
        results = await asyncio.gather(*tasks)
        return {g.id: stories for g, stories in zip(game_configs, results)}


async def _failed_judge_placeholder() -> JudgeResult:
    """Placeholder JudgeResult used when story generation itself failed."""
    return JudgeResult(
        passed=False,
        failed_criteria=["GENERATION_RETURNED_NO_STORY"],
        retry_hint="Generation produced no story — try again.",
    )
