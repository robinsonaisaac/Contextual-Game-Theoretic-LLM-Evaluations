# game_theory_llm/models.py
from dataclasses import dataclass
from typing import List, Tuple
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
    world_type: str
    actor_type: str
    prompt: str = None
    decision: str = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()