# examples/basic_generation.py
"""Basic story generation example."""

import asyncio
import logging

from dotenv import load_dotenv

from game_theory_llm import LLMClient, PayoffMatrix, StoryGenerator, get_game

load_dotenv()
logging.basicConfig(level=logging.INFO)


async def main():
    # Create a Prisoner's Dilemma matrix
    matrix = PayoffMatrix([
        (0, 0),       # Both choose A
        (100, -50),   # Agent 1 A, Agent 2 B
        (-50, 100),   # Agent 1 B, Agent 2 A
        (75, 75),     # Both choose B
    ])

    # Initialize client and generator
    client = LLMClient()
    generator = StoryGenerator(client)

    # Generate stories using a raw payoff matrix (original approach)
    stories = await generator.generate_stories(
        payoff_matrix=matrix,
        topic="mv_pharma_pro",
        actor_type="enemies",
        observability="private",
        power_dynamic="symmetric",
        n_stories=2,
    )

    # Print results
    for i, story in enumerate(stories, 1):
        print(f"\nStory {i} (raw matrix):")
        print(story.content)
        print(f"\nDecision: {story.decision}")
        print("-" * 80)

    # Generate stories using a named game config (Stag Hunt)
    # The game_config provides the matrix AND semantic decision labels
    stag_hunt = get_game("stag_hunt")
    stories_sh = await generator.generate_stories(
        payoff_matrix=stag_hunt.matrix,  # ignored when game_config is set
        topic="mv_pharma_pro",
        actor_type="allies",
        observability="private",
        power_dynamic="symmetric",
        n_stories=2,
        game_config=stag_hunt,
    )

    for i, story in enumerate(stories_sh, 1):
        print(f"\nStory {i} (Stag Hunt):")
        print(story.content)
        print(f"\nDecision: {story.decision} | Game: {story.game_type}")
        print("-" * 80)

    # Generate stories using multi-turn conversation mode
    # Each story uses a 2-turn conversation: context engagement → decision
    stories_mt = await generator.generate_stories(
        payoff_matrix=matrix,
        topic="mv_pharma_pro",
        actor_type="allies",
        observability="private",
        power_dynamic="symmetric",
        n_stories=2,
        conversation_mode="multi_turn",
    )

    for i, story in enumerate(stories_mt, 1):
        print(f"\nStory {i} (Multi-turn):")
        print(story.content)
        print(f"\nDecision: {story.decision} | Mode: {story.conversation_mode}")
        if story.conversation_history:
            print(f"Conversation turns: {len(story.conversation_history)}")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
