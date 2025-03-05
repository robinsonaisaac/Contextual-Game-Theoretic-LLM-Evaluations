#TODO: Add EV graph 
 
# game_theory_llm/api.py
from together import AsyncTogether
import os
from typing import Optional
from dotenv import load_dotenv
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from datetime import datetime, timedelta
import asyncio
import time
import random


# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Story:
    """Story data class."""
    content: str
    topic: str
    world_type: str
    actor_type: str

@dataclass
class PayoffMatrix:
    """Payoff matrix data class."""
    matrix: List[List[float]]

@dataclass
class AnalysisResult:
    """Results from analyzing decisions."""
    stories: List[Story]
    decisions: Dict[str, List[Optional[str]]]
    summaries: Dict[str, List[str]]
    proportions: Dict[str, Dict[str, float]]
    by_topic: Dict[str, Dict[str, Dict[str, float]]]
    by_world: Dict[str, Dict[str, Dict[str, float]]]
    by_actor: Dict[str, Dict[str, Dict[str, float]]]

    # NEW FIELD: store the time when analysis was created
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

load_dotenv()


class APIClient:
    """Mock API client for demonstration."""
    def __init__(self, api_key: Optional[str] = None, api_key_2: Optional[str] = None):
        # Primary API client setup
        self.api_key = api_key or os.getenv('TOGETHER_API_KEY')
        self.api_key_2 = api_key_2 or os.getenv('TOGETHER_API_KEY_2')
        
        if not self.api_key:
            raise ValueError(
                "Primary API key must be provided either through:\n"
                "1. TOGETHER_API_KEY environment variable\n"
                "2. .env file with TOGETHER_API_KEY=your-key\n"
                "3. Directly to APIClient(api_key='your-key')"
            )
            
        # Initialize both Together clients
        self.client = AsyncTogether(api_key=self.api_key)
        self.client_2 = AsyncTogether(api_key=self.api_key_2) if self.api_key_2 else None
        
        # Initialize other clients
        self.claude_client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Rate limiting setup for both Together AI instances (70% of limits for safety)
        self.requests_per_minute = 1400  # 21 QPS * 60
        self.tokens_per_minute = 200000
        
        # Separate tracking for each API key
        self.rate_limits = {
            'primary': {
                'request_timestamps': [],
                'token_usage': [],
                'last_request_time': 0
            },
            'secondary': {
                'request_timestamps': [],
                'token_usage': [],
                'last_request_time': 0
            }
        }
        self.min_request_interval = 1/20

    async def _enforce_rate_limits(self, api_key: str, tokens_to_use: int = 0):
        """Enforce rate limits for both requests and tokens for specified API key."""
        limits = self.rate_limits[api_key]
        current_time = time.time()
        minute_ago = current_time - 60

        # Enforce minimum interval between requests
        time_since_last = current_time - limits['last_request_time']
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        # Clean up old timestamps
        limits['request_timestamps'] = [ts for ts in limits['request_timestamps'] if ts > minute_ago]
        limits['token_usage'] = [(ts, tokens) for ts, tokens in limits['token_usage'] if ts > minute_ago]

        # Check request rate limit
        if len(limits['request_timestamps']) >= self.requests_per_minute:
            sleep_time = max(limits['request_timestamps'][0] - minute_ago + 0.1, 0.1)
            return False, sleep_time

        # Check token rate limit
        current_token_usage = sum(tokens for _, tokens in limits['token_usage'])
        if current_token_usage + tokens_to_use > self.tokens_per_minute:
            sleep_time = max(limits['token_usage'][0][0] - minute_ago + 0.1, 0.1)
            return False, sleep_time

        # Add new request and token usage
        limits['request_timestamps'].append(current_time)
        if tokens_to_use > 0:
            limits['token_usage'].append((current_time, tokens_to_use))
        
        limits['last_request_time'] = current_time
        return True, 0

    async def _try_llama_request(self, prompt: str, client, api_key_type: str):
        """Attempt a request with specified client and handle rate limits."""
        estimated_input_tokens = len(prompt) // 2
        can_proceed, wait_time = await self._enforce_rate_limits(api_key_type, estimated_input_tokens)
        
        if not can_proceed:
            return False, wait_time, None
            
        try:
            response = await client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16000,
                temperature=0.0
            )
            
            # Update token usage with actual usage
            actual_tokens = response.usage.total_tokens
            limits = self.rate_limits[api_key_type]
            if limits['token_usage']:
                limits['token_usage'][-1] = (limits['token_usage'][-1][0], actual_tokens)
            
            return True, 0, response
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                return False, 3, None  # Base wait time for rate limits
            raise

    async def generate(self, prompt: str, model: str = "all") -> Dict[str, str]:
        """Generate text using multiple models or a specific model."""
        responses = {}
        max_retries = 10
        base_wait = 3
        
        def should_retry(error: Exception) -> bool:
            """Helper function to determine if we should retry based on error type."""
            error_message = str(error).lower()
            return any(msg in error_message for msg in [
                "rate_limit", 
                "429", 
                "503", 
                "service unavailable",
                "capacity",
                "timeout",
                "server error"
            ])

        try:
            # Llama (Together AI)
            if model in ["all", "llama"]:
                for attempt in range(max_retries):
                    # Try primary API first
                    success, wait_time, response = await self._try_llama_request(prompt, self.client, 'primary')
                    
                    if success:
                        responses["llama"] = response.choices[0].message.content
                        break
                    
                    # If primary failed and we have a secondary API, try that
                    if not success and self.client_2:
                        success, secondary_wait, response = await self._try_llama_request(
                            prompt, self.client_2, 'secondary'
                        )
                        if success:
                            responses["llama"] = response.choices[0].message.content
                            break
                        wait_time = min(wait_time, secondary_wait)
                    
                    # If both failed and we haven't maxed out retries, wait and try again
                    if attempt < max_retries - 1:
                        wait_time = (base_wait ** (attempt + 1)) + random.uniform(1, 5)
                        logger.warning(
                            f"Service unavailable or rate limit hit on both Llama APIs, "
                            f"waiting {wait_time:.2f} seconds before retry (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries ({max_retries}) reached for Llama API")
                        responses["llama"] = None

            # Claude
            if model in ["all", "claude"]:
                for attempt in range(max_retries):
                    try:
                        claude_response = await self.claude_client.messages.create(
                            model=" ",
                            max_tokens=4096,
                            temperature=0.0,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        responses["claude"] = claude_response.content[0].text
                        break
                    except Exception as e:
                        if should_retry(e) and attempt < max_retries - 1:
                            wait_time = (base_wait ** (attempt + 1)) + random.uniform(1, 5)
                            logger.warning(
                                f"Service unavailable or rate limit hit on Claude API, "
                                f"waiting {wait_time:.2f} seconds before retry (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Error with Claude model: {str(e)}")
                            responses["claude"] = None
                            break

            # GPT-4
            if model in ["all", "gpt4"]:
                for attempt in range(max_retries):
                    try:
                        gpt_response = await self.openai_client.chat.completions.create(
                            model="gpt-4o-2024-11-20",
                            max_tokens=4096,
                            temperature=0.0,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        responses["gpt4"] = gpt_response.choices[0].message.content
                        break
                    except Exception as e:
                        if should_retry(e) and attempt < max_retries - 1:
                            wait_time = (base_wait ** (attempt + 1)) + random.uniform(1, 5)
                            logger.warning(
                                f"Service unavailable or rate limit hit on GPT-4 API, "
                                f"waiting {wait_time:.2f} seconds before retry (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Error with GPT-4 model: {str(e)}")
                            responses["gpt4"] = None
                            break

            return responses

        except Exception as e:
            logger.error(f"Error in generate function: {str(e)}")
            return {"llama": None, "claude": None, "gpt4": None}
    
    async def process_story(self, story: Story) -> Dict[str, str]:
        """Process a single story with each LLM to extract decisions."""
        logger.debug(f"Processing story (ID: {id(story)})")
        prompt = f"""
            {story.content}

            Analyze the scenario and provide:
            1. Your analysis of the potential outcomes
            2. A clear decision (A or B)

            Format your response as:
            <analysis>Your detailed analysis here</analysis>
            <decision>A or B</decision>
        """
        try:
            logger.debug("Sending prompt to API client")
            responses = await self.generate(prompt)  # returns dict with model responses
            logger.debug("Received responses from API clients")
            
            # Extract decisions for each model
            decisions = {}
            for model, response in responses.items():
                if response:  # Check if response string is valid
                    decision = self._extract_decision(response)
                    decisions[model] = decision
                    response_key = model+"_response"
                    decisions[response_key] = response
                else:
                    logger.warning(f"No response received from {model}")
                    decisions[model] = None
                    response_key = model+"_response"
                    decisions[response_key] = response
            
            logger.info(f"Extracted decisions: {decisions}")
            return decisions
        
        except Exception as e:
            logger.error(f"Error processing story {id(story)}: {str(e)}", exc_info=True)
            return {"llama": None, "claude": None, "gpt4": None}

    def _extract_decision(self, text: str) -> Optional[str]:
        """Parse out the decision tag with multiple possible formats.
        
        Handles multiple decision patterns:
        1. A/B format: <decision>A</decision> or <decision>B</decision>
        2. 1/2 format: <decision>1</decision> or <decision>2</decision>
        3. */**: <decision>*</decision> or <decision>**</decision>
        4. yellow/green: <decision>yellow</decision> or <decision>green</decision>
        5. #/&: <decision>#</decision> or <decision>&</decision>
        
        Returns:
            str: 'A' or 'B' if a valid decision is found
            None: if no valid decision pattern is found
        """
        patterns = [
            # Pattern 1: A/B
            (r'<decision>\s*([AB])\s*</decision>', {'A': 'A', 'B': 'B'}),
            # Pattern 2: 1/2
            (r'<decision>\s*([12])\s*</decision>', {'1': '1', '2': '2'}),
            # Pattern 3: */**
            (r'<decision>\s*(\*{1,2})\s*</decision>', {'&': '&', '&&': '&&'}),
            # Pattern 4: yellow/green
            (r'<decision>\s*(yellow|green)\s*</decision>', {'yellow': 'yellow', 'green': 'green'}),
            # Pattern 5: #/&
            (r'<decision>\s*([#&])\s*</decision>', {'#': '#', '&': '&'})
        ]
        
        for pattern, mapping in patterns:
            match = re.search(pattern, text)
            if match:
                decision = match.group(1)
                mapped_decision = mapping.get(decision)
                if mapped_decision:
                    logger.debug(f"Extracted decision: {decision} -> mapped to: {mapped_decision}")
                    return mapped_decision
        
        logger.warning("Could not parse <decision> from response")
        return None
    
    async def analyze_stories(
        self,
        stories: List[Story],
        payoff_matrix: Optional[PayoffMatrix] = None,
        df_existing=None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> AnalysisResult:
        """Analyze a collection of stories, returning multi-model decisions & distributions."""
        logger.info(f"Starting analysis of {len(stories)} stories")
        
        if df_existing is not None:
            # If an existing DataFrame is provided, assume decisions are also present
            logger.info("Existing dataframe passed, using its decisions")
            df = df_existing
            decisions = {
                model: list(df[f'decision_{model}']) 
                for model in ['llama', 'claude', 'gpt4']
            }
        else:
            # Otherwise, gather decisions from each model
            decisions = {'llama': [], 'claude': [], 'gpt4': [], 'llama_response': [], 'claude_response': [], 'gpt4_response': []}
            for i, story in enumerate(stories, 1):
                logger.debug(f"Processing story {i}/{len(stories)}")
                decision_dict = await self.process_story(story)
                for model, decision in decision_dict.items():
                    # response_key = model+"_response"
                    decisions[model].append(decision)
                if progress_callback:
                    progress_callback(1)  # Update progress by 1
            
            # Create a DataFrame for further analysis
            df = pd.DataFrame({
                'topic': [s.topic for s in stories],
                'world_type': [s.world_type for s in stories],
                'actor_type': [s.actor_type for s in stories]
            })
            for model, model_decisions in decisions.items():
                if "response" in model:
                    df[f'response_{model}'] = model_decisions
                else:
                    df[f'decision_{model}'] = model_decisions

        # Prepare data structures to hold analysis
        proportions = {}
        by_topic = {}
        by_world = {}
        by_actor = {}
        summaries = {}

        # Calculate distributions per model
        for model in ['llama', 'claude', 'gpt4']:
            decision_col = f'decision_{model}'
            # Overall proportions
            proportions[model] = df[decision_col].value_counts(normalize=True).to_dict()
            # By topic
            by_topic[model] = {}
            for tp in df['topic'].unique():
                sub_df = df[df['topic'] == tp]
                by_topic[model][tp] = sub_df[decision_col].value_counts(normalize=True).to_dict()
            # By world
            by_world[model] = {}
            for wld in df['world_type'].unique():
                sub_df = df[df['world_type'] == wld]
                by_world[model][wld] = sub_df[decision_col].value_counts(normalize=True).to_dict()
            # By actor
            by_actor[model] = {}
            for act in df['actor_type'].unique():
                sub_df = df[df['actor_type'] == act]
                by_actor[model][act] = sub_df[decision_col].value_counts(normalize=True).to_dict()
            
            # Summaries for each model
            summaries[model] = self._generate_summaries(df, payoff_matrix, model)

        logger.info("Analysis complete")
        return AnalysisResult(
            stories=stories,
            decisions=decisions,
            summaries=summaries,
            proportions=proportions,
            by_topic=by_topic,
            by_world=by_world,
            by_actor=by_actor
        )

    def _generate_summaries(self, df: pd.DataFrame, payoff_matrix: Optional[PayoffMatrix], model: str) -> List[str]:
        """Generate textual summaries for each model's decisions."""
        summaries = []
        decision_col = f'decision_{model}'
        
        # Overall distribution
        dist = df[decision_col].value_counts(normalize=True).to_dict()
        summaries.append(f"Overall Decision Distribution: {dist}")
        
        # By category
        for cat in ['topic', 'world_type', 'actor_type']:
            for val in df[cat].unique():
                sub_df = df[df[cat] == val]
                sub_dist = sub_df[decision_col].value_counts(normalize=True).to_dict()
                summaries.append(f"Distribution by {cat}={val}: {sub_dist}")
        
        # Game-theoretic analysis if payoff matrix was provided
        if payoff_matrix:
            prob_a = df[decision_col].eq('A').mean()
            summaries.append(f"Probability of choosing A: {prob_a:.2f}")
            expected_utility = self._calculate_expected_utility(prob_a, payoff_matrix)
            summaries.append(f"Expected utility: {expected_utility:.2f}")
        
        return summaries

    def _calculate_expected_utility(self, prob_a: float, matrix: PayoffMatrix) -> float:
        """Calculate expected utility given probability of choosing A in a 2x2 scenario."""
        prob_b = 1 - prob_a
        # matrix.matrix is assumed to be 4 rows of length 2, e.g. PD matrix
        # Row 0: both choose A
        # Row 1: A,B
        # Row 2: B,A
        # Row 3: B,B
        utility = (
            prob_a * prob_a * matrix.matrix[0][0]     # Both A
            + prob_a * prob_b * matrix.matrix[1][0]   # A,B
            + prob_b * prob_a * matrix.matrix[2][0]   # B,A
            + prob_b * prob_b * matrix.matrix[3][0]   # Both B
        )
        return utility
    
    def visualize_results(self, results: AnalysisResult, save_path: Optional[str] = None):
        """Create basic visualizations of the analysis results for each model."""
        logger.info("Creating visualizations")
        import matplotlib.pyplot as plt
        import pandas as pd
        
        for model in ['llama', 'claude', 'gpt4']:
            # Overall decision distribution
            plt.figure(figsize=(8, 5))
            model_decisions = results.decisions[model]
            pd.Series(model_decisions).value_counts().plot(kind='bar', color='skyblue')
            plt.title(f"Overall Decision Distribution - {model}")
            plt.xlabel("Decision")
            plt.ylabel("Count")
            if save_path:
                plt.savefig(f"{save_path}/overall_dist_{model}.png")
            plt.close()
        logger.info("Visualization creation complete")
            