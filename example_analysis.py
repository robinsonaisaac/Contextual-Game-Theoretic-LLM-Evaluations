# example_analysis.py
import asyncio
import os
import pickle
import json
from game_theory_llm import PayoffMatrix, StoryGenerator, APIClient
from game_theory_llm.analyzer import StoryAnalyzer

async def main():
    # Create payoff matrix
    matrix = PayoffMatrix([
        (0, 0),       # Both choose A
        (100, -50),   # Agent 1 A, Agent 2 B
        (-50, 100),   # Agent 1 B, Agent 2 A
        (75, 75)      # Both choose B
    ])

    # Initialize components
    api_client = APIClient()
    generator = StoryGenerator(api_client)
    analyzer = StoryAnalyzer(api_client)
    
    # Generate stories
    print("Generating stories...")
    stories = await generator.generate_stories(
        payoff_matrix=matrix,
        topic="International Business",
        world_type="real world",
        actor_type="allies",
        n_stories=15  # Adjust as needed
    )
    
    # Save stories using pickle
    print("\nSaving stories to pickle file...")
    with open('all_stories.pkl', 'wb') as f:
        pickle.dump(stories, f)
    print("Stories saved to 'all_stories.pkl'")
    
    # Save stories in human-readable format
    print("\nSaving stories to readable file...")
    readable_stories = []
    for i, story in enumerate(stories, 1):
        readable_story = {
            'story_number': i,
            'content': story.content,
            'topic': story.topic,
            'world_type': story.world_type,
            'actor_type': story.actor_type,
            'prompt': story.prompt,
            'decision': story.decision,
            'timestamp': story.timestamp.isoformat() if story.timestamp else None
        }
        readable_stories.append(readable_story)
    
    # Save as JSON with nice formatting
    with open('all_stories_readable.json', 'w', encoding='utf-8') as f:
        json.dump(readable_stories, f, indent=2, ensure_ascii=False)
    
    # Save as plain text for easiest reading
    with open('all_stories.txt', 'w', encoding='utf-8') as f:
        for i, story in enumerate(stories, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"Story {i}\n")
            f.write(f"Topic: {story.topic}\n")
            f.write(f"World Type: {story.world_type}\n")
            f.write(f"Actor Type: {story.actor_type}\n")
            f.write(f"Decision: {story.decision}\n")
            f.write(f"Timestamp: {story.timestamp}\n")
            f.write(f"\nPrompt Used:\n{story.prompt}\n")
            f.write(f"\nContent:\n{story.content}\n")
    
    print("Stories saved to 'all_stories_readable.json' and 'all_stories.txt'")
    
    print(f"Generated {len(stories)} stories")
    for story in stories:
        print(story.content[:100])
    
    # Analyze stories
    print("\nAnalyzing decisions...")
    results = await analyzer.api_client.analyze_stories(stories, matrix)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Game theory result summaries: {results.summaries}")
    print(f"Overall Proportions: {results.proportions}")
    print("\nBreakdown by Topic:")
    for topic, dist in results.by_topic.items():
        print(f"{topic}: {dist}")
    print("\nBreakdown by World Type:")
    for world, dist in results.by_world.items():
        print(f"{world}: {dist}")
    print("\nBreakdown by Actor Type:")
    for actor, dist in results.by_actor.items():
        print(f"{actor}: {dist}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    os.makedirs("analysis_output", exist_ok=True)
    analyzer.api_client.visualize_results(results, save_path="analysis_output")
    print("Visualizations saved to 'analysis_output' directory")

if __name__ == "__main__":
    asyncio.run(main())