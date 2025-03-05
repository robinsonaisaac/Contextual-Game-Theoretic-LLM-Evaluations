#!/usr/bin/env python3
"""
Story analyzer module for game theory analysis.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .api import APIClient

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
    decisions: List[str]
    summaries: List[str]
    proportions: Dict[str, float]
    by_topic: Dict[str, Dict[str, float]]
    by_world: Dict[str, Dict[str, float]]
    by_actor: Dict[str, Dict[str, float]]


class StoryAnalyzer:
    """Analyzes stories and decisions."""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        logger.info("Initializing StoryAnalyzer with API client")
        