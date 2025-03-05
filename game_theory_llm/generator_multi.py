# game_theory_llm/generator.py
from typing import List, Tuple, Optional
import re
import logging
from .models import PayoffMatrix, Story
from .api import APIClient
from .config import WORLD_DICT, ACTOR_TYPES, TOPICS
import asyncio
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatchGenerationResult:
    """Results from a batch of story generation."""
    stories: List[Story]
    summaries: List[str]
    unique_prompt: str

class StoryGenerator:
    """Generates stories from game theory scenarios."""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        logger.info("StoryGenerator initialized")

    async def generate_story_summary(self, story: str) -> str:
        """Generate a concise, structured summary of a story using the API."""
        logger.debug("Generating summary for story")
        prompt = """Analyze this story and create a single, comprehensive sentence that captures:
        1. The main character(s) and their defining traits
        2. The primary setting/location
        3. The core conflict or goal
        4. The most important plot development

        Story to summarize:
        {story}"""

        try:
            summary = await self.api_client.generate(prompt.format(story=story))
            logger.debug(f"Generated summary: {summary[:100]}...")
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"A story about {story[:100]}..."

    def generate_unique_prompt(self, summaries: List[str]) -> str:
        """Generate a prompt that helps ensure unique stories based on previous summaries."""
        logger.debug(f"Generating unique prompt from {len(summaries)} summaries")
        if not summaries:
            return ""
            
        all_summaries = "\n".join(f"{i+1}. {summary}" for i, summary in enumerate(summaries))
        return f"""Previous story summaries (avoid reusing elements from these):
        {all_summaries}

        Generate completely new stories with different characters, settings, and plots."""

    def create_query(
        self,
        matrix: PayoffMatrix,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = ""
    ) -> str:
        """Create a prompt using the original format."""
        logger.debug("Creating query prompt")
        prompt = f"""Write {10} unique stories about a scenario involving two agents and their possible actions:

Matrix: {matrix.format_matrix()}

Topic: {topic}
Relationship: {world_type} {actor_type}

Each agent has two decisions (A or B):
- Both choose A: {matrix.matrix[0]}
- Agent 1 A, Agent 2 B: {matrix.matrix[1]}
- Agent 1 B, Agent 2 A: {matrix.matrix[2]}
- Both choose B: {matrix.matrix[3]}

{WORLD_DICT[world_type]}

RELATIONSHIP DESCRIPTION:
{ACTOR_TYPES[actor_type]['description']}

EXAMPLES OF THIS TYPE OF RELATIONSHIP INCLUDE:
{ACTOR_TYPES[actor_type]['types']}


Rules:
1. Do not under any cirumstance mention that this is a game
2. Label decisions as A or B
3. Don't show explicit payoffs
4. Show how outcomes depend on both agents
5. Surround each story with <story></story> tags

{unique_prompt}

End each story with: 
<toadd>
The name of agent 1 is <name> <agent1> </name> 
The name of agent 1 is <name> <agent2> </name> 
<justification></justification>
<decision>A or B</decision>
</toadd>"""
        
        logger.debug(f"Created prompt of length {len(prompt)}")
        return prompt

    async def generate_batch(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        world_type: str,
        actor_type: str,
        unique_prompt: str = ""
    ) -> BatchGenerationResult:
        """Generate a batch of stories with summaries."""
        logger.info("Generating batch of stories")
        prompt = self.create_query(payoff_matrix, topic, world_type, actor_type, unique_prompt)
        
        try:
            content = await self.api_client.generate(prompt)
            content = content["llama"]
            logger.debug(f"Generated content length: {len(content)}")
            
            # Extract stories
            story_pattern = r'<story>(.*?)</story>'
            raw_stories = re.findall(story_pattern, content, re.DOTALL)
            logger.info(f"Extracted {len(raw_stories)} stories from response")
            
            if not raw_stories:
                logger.warning("No stories found in generated content")
                logger.debug(f"Content preview: {content[:500]}...")
                return BatchGenerationResult([], [], "")
            
            stories = []
            for story_content in raw_stories:
                decision = self._extract_decision(story_content)
                story = Story(
                    content=story_content.strip(),
                    topic=topic,
                    world_type=world_type,
                    actor_type=actor_type,
                    decision=decision
                )
                stories.append(story)
            
            # Generate summaries
            summaries = await asyncio.gather(
                *[self.generate_story_summary(story.content) for story in stories]
            )
            
            new_unique_prompt = self.generate_unique_prompt(summaries)
            logger.info(f"Successfully generated batch with {len(stories)} stories")
            
            return BatchGenerationResult(stories, summaries, new_unique_prompt)
            
        except Exception as e:
            logger.error(f"Error generating batch: {str(e)}")
            return BatchGenerationResult([], [], "")

    async def generate_stories(
        self,
        payoff_matrix: PayoffMatrix,
        topic: str,
        world_type: str,
        actor_type: str,
        n_stories: int = 100,
        batch_size: int = 10
    ) -> List[Story]:
        """Generate multiple stories in batches."""
        logger.info(f"Starting generation of {n_stories} stories in batches of {batch_size}")
        
        # Input validation
        if topic not in TOPICS:
            raise ValueError(f"Invalid topic. Must be one of: {TOPICS}")
        if world_type not in WORLD_DICT:
            raise ValueError(f"Invalid world type. Must be one of: {list(WORLD_DICT.keys())}")
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"Invalid actor type. Must be one of: {list(ACTOR_TYPES.keys())}")
        
        all_stories = []
        all_summaries = []
        unique_prompt = ""
        
        n_batches = (n_stories + batch_size - 1) // batch_size
        logger.info(f"Will generate {n_batches} batches")
        
        for batch_num in range(n_batches):
            logger.info(f"Generating batch {batch_num + 1}/{n_batches}")
            batch_result = await self.generate_batch(
                payoff_matrix, 
                topic, 
                world_type, 
                actor_type, 
                unique_prompt
            )
            
            if batch_result.stories:
                all_stories.extend(batch_result.stories)
                all_summaries.extend(batch_result.summaries)
                unique_prompt = batch_result.unique_prompt
                logger.info(f"Added {len(batch_result.stories)} stories from batch {batch_num + 1}")
            else:
                logger.warning(f"Batch {batch_num + 1} generated no stories")
            
            if len(all_stories) >= n_stories:
                logger.info(f"Reached target number of stories ({n_stories})")
                break
        
        logger.info(f"Generation complete. Generated {len(all_stories)} stories total")
        return all_stories[:n_stories]

    def _extract_decision(self, text: str) -> Optional[str]:
        """Extract decision from story."""
        decision_pattern = r'<decision>\s*([AB])\s*</decision>'
        match = re.search(decision_pattern, text)
        if match:
            logger.debug(f"Extracted decision: {match.group(1)}")
            return match.group(1)
        logger.warning("No decision found in story")
        return None