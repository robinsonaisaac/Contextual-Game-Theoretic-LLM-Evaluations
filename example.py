# example.py
import asyncio
import os
from game_theory_llm import PayoffMatrix, StoryGenerator, APIClient

async def main():
    # Create a Prisoner's Dilemma matrix
    matrix = PayoffMatrix([
        (0, 0),       # Both choose A
        (100, -50),   # Agent 1 A, Agent 2 B
        (-50, 100),   # Agent 1 B, Agent 2 A
        (75, 75)      # Both choose B
    ])

    # Initialize the API client
    api_client = APIClient()
    
    # Create story generator
    generator = StoryGenerator(api_client)
    
    # Generate stories
    stories = await generator.generate_stories(
        payoff_matrix=matrix,
        topic="international business",
        world_type="real world",
        actor_type="enemies",
        n_stories=2
    )
    
    # Print results
    for i, story in enumerate(stories, 1):
        print(f"\nStory {i}:")
        print(story.content)
        print(f"\nDecision: {story.decision}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())