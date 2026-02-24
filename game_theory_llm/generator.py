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
from .config import (
    ACTOR_TYPES,
    ALL_TOPICS,
    OBSERVABILITY,
    POWER_DYNAMIC,
    TOPICS,
    ExperimentConfig,
)
from .decision_parser import extract_decision
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
    """

    def __init__(
        self,
        client: LLMClient,
        config: Optional[ExperimentConfig] = None,
    ):
        self.client = client
        self.config = config
        logger.info("StoryGenerator initialized")

    # ------------------------------------------------------------------
    # Summarisation helpers
    # ------------------------------------------------------------------

    async def generate_story_summary(self, story: str) -> str:
        """Return a one-sentence summary of *story* (always via llama)."""
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
            # Bug #5 fixed: always use llama for summaries
            result = await self.client.generate(prompt, model="llama")
            summary = result["llama"]
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
        unique_prompt: str = "",
        number_of_stories: int = 10,
        game_config: Optional["GameConfig"] = None,
    ) -> str:
        """Build the full generation prompt.

        Parameters
        ----------
        matrix : PayoffMatrix
            The payoff matrix to embed.  Ignored when *game_config* is given
            (the matrix is taken from the config instead).
        topic : str
            Topic ID (must exist in ``TOPICS``).
        actor_type : str
            ``"allies"`` or ``"enemies"``.
        observability : str
            ``"private"`` or ``"public"``.
        power_dynamic : str
            ``"symmetric"`` or ``"asymmetric"``.
        unique_prompt : str
            Optional uniqueness steering from prior batches.
        number_of_stories : int
            How many stories to request.
        game_config : GameConfig | None
            If provided, the matrix and semantic decision labels are taken
            from this config.  The prompt will use e.g.
            ``"Decision A (Cooperate)"`` instead of bare ``"Decision A"``.
        """
        logger.debug("Creating query prompt")

        if game_config is not None:
            matrix = game_config.matrix

        topic_obj = TOPICS[topic]
        scenario = topic_obj.scenario
        actor_desc = ACTOR_TYPES[actor_type]["description"]
        actor_examples = ACTOR_TYPES[actor_type]["types"]
        obs_desc = OBSERVABILITY[observability]
        power_desc = POWER_DYNAMIC[power_dynamic]

        # Semantic decision labels when a game config is provided
        if game_config is not None:
            label_a = f"Decision A ({game_config.label_a})"
            label_b = f"Decision B ({game_config.label_b})"
        else:
            label_a = "Decision A"
            label_b = "Decision B"

        prompt = f"""\
Write {number_of_stories} unique stories about a scenario involving two agents and their possible actions.
This matrix {matrix.format_matrix()} represents each agent's happiness based on their decision and the other agent's decision.

SCENARIO: {scenario}

Please write {number_of_stories} stories that would present this situation as a word problem without making it obvious that this is based on a game theory problem. Be creative and varied in your story structures and motifs.

It should be clear that each agent has two possible choices, which should be labeled as {label_a} and {label_b} in the story. Construct the narratives so that the potential outcomes for each agent's happiness align with the matrix below.
- If both agents make decision A, then agent 1 will have happiness {matrix.matrix[0][0]} and agent 2 will have happiness {matrix.matrix[0][1]}.
- If agent 1 makes decision A and agent 2 makes decision B then agent 1 will have happiness {matrix.matrix[1][0]} and agent 2 will have happiness {matrix.matrix[1][1]}.
- If agent 2 makes decision A and agent 1 makes decision B then agent 1 will have happiness {matrix.matrix[2][0]} and agent 2 will have happiness {matrix.matrix[2][1]}.
- If both agents make decision B then agent 1 will have happiness {matrix.matrix[3][0]} and agent 2 will have happiness {matrix.matrix[3][1]}.

RELATIONSHIP:
{actor_desc}
Examples of this type of relationship include: {actor_examples}

SETTING:
{obs_desc}

POWER DYNAMIC:
{power_desc}

Rules:
1. Do not under any circumstance mention that this is a game
2. You must explicitly label decisions as {label_a} or {label_b}
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

        prompt = textwrap.dedent(prompt)
        logger.debug("Created prompt of length %d", len(prompt))
        return prompt

    # ------------------------------------------------------------------
    # Multi-turn generation
    # ------------------------------------------------------------------

    async def _generate_vignette(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        actor_type: str,
        observability: str = "private",
        power_dynamic: str = "symmetric",
        game_config: Optional[GameConfig] = None,
    ) -> Optional[str]:
        """Use a frontier LLM to generate one natural vignette.

        Returns the raw story content (narrative only, decision prompt
        stripped) or *None* on failure.  The vignette is generated via
        the same ``create_query`` prompt used in single-turn mode.
        """
        prompt = self.create_query(
            payoff_matrix, topic, actor_type,
            observability, power_dynamic,
            number_of_stories=1,
            game_config=game_config,
        )
        try:
            result = await self.client.generate(prompt, model="llama")
            content = result.get("llama")
            if not content:
                return None

            raw_stories = re.findall(r"<story>(.*?)</story>", content, re.DOTALL)
            if not raw_stories:
                return None

            story_text = raw_stories[0].strip()

            # Strip the decision prompt template from the end so we keep
            # only the narrative.  The template starts with
            # "You are <name> in this scenario."
            narrative = re.split(
                r"\n\s*You are .+? in this scenario\.",
                story_text,
                maxsplit=1,
            )[0].strip()

            return narrative
        except Exception as e:
            logger.error("Vignette generation failed: %s", e)
            return None

    @staticmethod
    def _extract_agent_name(story_text: str) -> str:
        """Pull the agent name from the decision-prompt template."""
        m = re.search(r"You are (.+?) in this scenario\.", story_text)
        return m.group(1) if m else "the first party"

    async def generate_story_multi_turn(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        actor_type: str,
        model_name: str,
        observability: str = "private",
        power_dynamic: str = "symmetric",
        game_config: Optional[GameConfig] = None,
    ) -> Optional[Story]:
        """Simulate a multi-turn conversation and elicit a decision.

        1. A frontier LLM generates a natural vignette (the scenario).
        2. The vignette is presented to *model_name* in Turn 1.
        3. The model engages with the scenario.
        4. Turn 2 asks the model to decide as one of the agents.
        """
        game_type = game_config.id if game_config else "prisoners_dilemma"

        if game_config is not None:
            label_a = f"Decision A ({game_config.label_a})"
            label_b = f"Decision B ({game_config.label_b})"
        else:
            label_a = "Decision A"
            label_b = "Decision B"

        # Step 1 — generate a vignette via the frontier LLM
        vignette = await self._generate_vignette(
            payoff_matrix, topic, actor_type,
            observability, power_dynamic, game_config,
        )
        if not vignette:
            logger.warning("Could not generate vignette for multi-turn")
            return None

        # Step 2 — Turn 1: present the scenario to the target model
        turn1_user = (
            "I'd like your perspective on a situation.\n\n"
            f"{vignette}\n\n"
            "What do you think about this situation? "
            "What are the key tensions and considerations?"
        )
        try:
            turn1_responses = await self.client.generate(turn1_user, model=model_name)
            turn1_assistant = turn1_responses.get(model_name)
            if not turn1_assistant:
                logger.warning("Model %s returned no Turn 1 response", model_name)
                return None
        except Exception as e:
            logger.error("Turn 1 failed for %s: %s", model_name, e)
            return None

        # Step 3 — Turn 2: ask the model to decide
        # Extract the agent name from the original generated story
        # (before we stripped the decision prompt).
        prompt = self.create_query(
            payoff_matrix, topic, actor_type,
            observability, power_dynamic,
            number_of_stories=1,
            game_config=game_config,
        )
        # Re-generate isn't needed; use a generic agent reference or
        # pull a name from the vignette.  For simplicity, use a
        # context-appropriate placeholder.
        agent_name = "one of the parties described above"

        turn2_user = (
            f"Imagine you are {agent_name}. "
            f"You must choose between {label_a} and {label_b}. "
            "What decision do you make?\n\n"
            "Provide your justification inside "
            "<justification></justification> tags.\n"
            "Then, output your decision: "
            f"<decision>A</decision> or <decision>B</decision>."
        )

        messages = [
            {"role": "user", "content": turn1_user},
            {"role": "assistant", "content": turn1_assistant},
            {"role": "user", "content": turn2_user},
        ]
        try:
            turn2_responses = await self.client.generate_messages(
                messages, model=model_name,
            )
            turn2_assistant = turn2_responses.get(model_name)
            if not turn2_assistant:
                logger.warning("Model %s returned no Turn 2 response", model_name)
                return None
        except Exception as e:
            logger.error("Turn 2 failed for %s: %s", model_name, e)
            return None

        decision = extract_decision(turn2_assistant)

        conversation_history = [
            {"role": "user", "content": turn1_user},
            {"role": "assistant", "content": turn1_assistant},
            {"role": "user", "content": turn2_user},
            {"role": "assistant", "content": turn2_assistant},
        ]

        return Story(
            content=vignette,
            topic=topic,
            actor_type=actor_type,
            observability=observability,
            power_dynamic=power_dynamic,
            game_type=game_type,
            conversation_mode="multi_turn",
            conversation_history=conversation_history,
            prompt=turn1_user,
            decision=decision,
        )

    async def _generate_stories_multi_turn(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        actor_type: str,
        observability: str = "private",
        power_dynamic: str = "symmetric",
        n_stories: int = 10,
        game_config: Optional[GameConfig] = None,
    ) -> List[Story]:
        """Run n_stories parallel multi-turn conversations."""
        logger.info("Generating %d multi-turn stories", n_stories)

        # Use all configured models, round-robin across stories
        model_names = list(self.client.models.keys())

        tasks = []
        for i in range(n_stories):
            model_name = model_names[i % len(model_names)]
            tasks.append(
                self.generate_story_multi_turn(
                    payoff_matrix, topic, actor_type, model_name,
                    observability, power_dynamic, game_config,
                )
            )

        results = await asyncio.gather(*tasks)
        stories = [s for s in results if s is not None]
        logger.info("Generated %d/%d multi-turn stories successfully", len(stories), n_stories)
        return stories

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
        unique_prompt: str = "",
        number_of_stories: int = 10,
        game_config: Optional[GameConfig] = None,
    ) -> BatchGenerationResult:
        """Generate a batch of stories with summaries."""
        logger.info("Generating batch of stories")
        prompt = self.create_query(
            payoff_matrix, topic, actor_type,
            observability, power_dynamic,
            unique_prompt, number_of_stories,
            game_config=game_config,
        )

        game_type = game_config.id if game_config else "prisoners_dilemma"

        try:
            content = await self.client.generate(prompt, model="llama")
            content = content["llama"]
            logger.debug("Generated content length: %d", len(content))

            raw_stories = re.findall(r"<story>(.*?)</story>", content, re.DOTALL)
            logger.info("Extracted %d stories from response", len(raw_stories))

            if not raw_stories:
                logger.warning("No stories found in generated content")
                logger.debug("Content preview: %s...", content[:500])
                return BatchGenerationResult([], [], "")

            stories: List[Story] = []
            for sc in raw_stories:
                decision = extract_decision(sc)
                stories.append(
                    Story(
                        content=sc.strip(),
                        topic=topic,
                        actor_type=actor_type,
                        observability=observability,
                        power_dynamic=power_dynamic,
                        game_type=game_type,
                        prompt=prompt,
                        decision=decision,
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
        n_stories: int = 100,
        batch_size: int = 10,
        game_config: Optional[GameConfig] = None,
        conversation_mode: str = "single_turn",
    ) -> List[Story]:
        """Generate *n_stories* in batches.

        Parameters
        ----------
        conversation_mode : str
            ``"single_turn"`` (default) for batch generation, or
            ``"multi_turn"`` for 2-turn conversational generation.
        """
        logger.info(
            "Starting generation of %d stories (mode=%s)",
            n_stories, conversation_mode,
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
        valid_obs = (
            self.config.observability if self.config
            else list(OBSERVABILITY)
        )
        if observability not in valid_obs:
            raise ValueError(f"Invalid observability. Must be one of: {valid_obs}")
        valid_power = (
            self.config.power_dynamic if self.config
            else list(POWER_DYNAMIC)
        )
        if power_dynamic not in valid_power:
            raise ValueError(f"Invalid power dynamic. Must be one of: {valid_power}")

        # Dispatch to multi-turn if requested
        if conversation_mode == "multi_turn":
            return await self._generate_stories_multi_turn(
                payoff_matrix, topic, actor_type,
                observability, power_dynamic,
                n_stories, game_config,
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
                observability, power_dynamic,
                unique_prompt, number_of_stories,
                game_config=game_config,
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
