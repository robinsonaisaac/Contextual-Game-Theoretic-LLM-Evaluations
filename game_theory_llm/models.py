# game_theory_llm/models.py
"""All data classes for the game_theory_llm package (single source of truth)."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime


@dataclass
class PayoffMatrix:
    """Represents a 2x2 payoff matrix for game theory scenarios."""
    matrix: List[Tuple[int, int]]

    def __post_init__(self):
        if len(self.matrix) != 4:
            raise ValueError("Payoff matrix must contain exactly 4 scenarios")

    def format_matrix(self) -> str:
        """Returns a formatted string representation of the payoff matrix."""
        return f"""
┌──────────────┬─────────────┬─────────────┐
│ Agent 1 ↓    │  Agent 2 →  │             │
├──────────────┼─────────────┼─────────────┤
│              │      A      │      B      │
├──────────────┼─────────────┼─────────────┤
│      A       │  {self.matrix[0][0]}, {self.matrix[0][1]}  │  {self.matrix[1][0]}, {self.matrix[1][1]}  │
├──────────────┼─────────────┼─────────────┤
│      B       │  {self.matrix[2][0]}, {self.matrix[2][1]}  │  {self.matrix[3][0]}, {self.matrix[3][1]}  │
└──────────────┴─────────────┴─────────────┘
"""


@dataclass
class Story:
    """A generated story and its metadata."""
    content: str
    topic: str
    actor_type: str
    observability: str = "private"
    power_dynamic: str = "symmetric"
    game_type: str = "prisoners_dilemma"
    conversation_mode: str = "single_turn"
    conversation_history: Optional[List[Dict[str, str]]] = None
    prompt: str = None
    decision: str = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AnalysisResult:
    """Results from analyzing decisions across multiple models."""
    stories: List[Story]
    decisions: Dict[str, List[Optional[str]]]
    summaries: Dict[str, List[str]]
    proportions: Dict[str, Dict[str, float]]
    by_topic: Dict[str, Dict[str, Dict[str, float]]]
    by_observability: Dict[str, Dict[str, Dict[str, float]]]
    by_power: Dict[str, Dict[str, Dict[str, float]]]
    by_actor: Dict[str, Dict[str, Dict[str, float]]]
    by_game: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BatchGenerationResult:
    """Results from a batch of story generation."""
    stories: List[Story]
    summaries: List[str]
    unique_prompt: str
