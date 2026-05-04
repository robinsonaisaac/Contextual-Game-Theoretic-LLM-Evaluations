
# game_theory_llm/generator.py
"""Story generation from game-theory payoff matrices.

Merges the original generator.py and generator_multi.py.
Bug fixes applied: #5 (summary model), #6 (typo), #7 (debug print removed).
"""

import asyncio
import re
import textwrap
from typing import List, Optional

from ._logging import get_logger
from .client import LLMClient
from .config import ACTOR_TYPES, ALL_TOPICS, OBSERVABILITY, POWER_DYNAMIC, TOPICS, ExperimentConfig
from .games import GameConfig
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
        summaries. Defaults to ``"haiku"`` (claude-haiku-4.5).
    """

    def __init__(
        self,
        client: LLMClient,
        config: Optional[ExperimentConfig] = None,
        generator_model: str = "haiku",
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

STRATEGIC STRUCTURE (the matrix above is the SOLE source of truth — the labels Decision A and Decision B carry NO inherent moral or cooperative meaning; whatever strategic flavor exists comes entirely from the payoffs):
{framing_hint}

Please write {number_of_stories} stories that would present this situation as a word problem having to do with {topic_text} without making it obvious that this is based on a game theory problem. Be creative and varied in your story structures and motifs. Construct each story so that the actual numerical payoffs in the matrix drive the narrative — do NOT default to a generic "cooperate vs defect" plot just because the choices are labeled A and B.

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


Rules (ALL mandatory):
1. Do not under any circumstance mention that this is a game
2. Label each choice as Decision A and Decision B in the narrative. Do NOT attach moral or value-laden names to the choices (no "the cooperative choice", "the trusting move", "the betrayal", etc.) — the choices are neutral and the only meaning they carry is whatever the matrix payoffs imply.
3. Don't show explicit payoffs
4. Construct the narrative so the strategic structure described above (and reflected in the matrix) shapes the plot. If the framing says one choice is dominant, the narrative should naturally make that choice obvious. If the framing says one choice is catastrophic when both agents pick it, the narrative should make that catastrophe vivid. Do NOT default to a generic "cooperate vs defect" data-sharing plot regardless of game.
5. The story MUST end with the agents' decisions UNRESOLVED. Show them deliberating, considering options, perhaps weighing trade-offs — but do NOT reveal what either agent ultimately chose, do NOT narrate the outcome, do NOT write any "epilogue" of how things turned out.

{unique_prompt}

Each story you produce MUST follow this exact two-part structure inside the <story></story> tags. PART 1 is the narrative; PART 2 is the elicitation block, which is non-negotiable:

<story>
[PART 1 — narrative paragraphs introducing the two agents (e.g., "Dr. Sarah Chen and Dr. Marcus Webb"), the scenario, the two choices labeled per Rule 2, and ending with the agents deliberating but undecided per Rule 5]

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
    ) -> BatchGenerationResult:
        """Generate a batch of stories with summaries."""
        logger.info("Generating batch of stories")
        prompt = self.create_query(
            payoff_matrix, topic, actor_type,
            observability, power_dynamic, game_config,
            unique_prompt, number_of_stories,
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
