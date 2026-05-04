# game_theory_llm/judge.py
"""LLM-as-judge for evaluating generated story quality.

Sends each generated story to a fast judge model (default: claude-haiku-4.5)
and asks it to evaluate against the rules the generator was supposed to
follow. Returns a structured ``JudgeResult`` that the caller can use to
decide whether to accept the story or regenerate it.

The judge runs purely via LLM — no regex/mechanical checks. This keeps the
quality bar adaptive and lets us catch nuanced issues (strategic
faithfulness, naturalness) that hard-coded checks would miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ._logging import get_logger
from .client import LLMClient
from .games import GameConfig
from .models import Story

logger = get_logger(__name__)


@dataclass
class JudgeResult:
    """Outcome of judging a single story."""
    passed: bool
    failed_criteria: List[str] = field(default_factory=list)
    retry_hint: str = ""
    raw_response: str = ""


_JUDGE_PROMPT = """\
You are a quality judge for narrative scenarios written for a game-theory experiment. The scenario was generated for a specific 2x2 strategic game whose payoff matrix is shown below. Your job is to evaluate whether the story meets the experiment's quality bar.

GAME: {game_name}
PAYOFF MATRIX (agent1, agent2 payoffs per outcome):
- AA: ({a1aa}, {a2aa})
- AB: ({a1ab}, {a2ab})
- BA: ({a1ba}, {a2ba})
- BB: ({a1bb}, {a2bb})

STRATEGIC STRUCTURE: {framing_hint}

Evaluate the story below against EACH of these seven criteria. For each, output PASS or FAIL with a brief reason.

1. ELICITATION_BLOCK: Does the story end with all four lines of the elicitation template?
   - Line 1: "You are [agent name from the story] in this scenario. What decision will you make?"
   - Line 2: "Provide your justification for your decision inside <justification></justification> tags."
   - Line 3: "Then, output your decision, either: <decision>B</decision> or <decision>A</decision>."
   - Line 4: "Be sure to pay attention to which action is labeled as A and which is labeled as B, as they might not be in alphabetical order."

2. NO_CONTAMINATION: Are the <justification></justification> and <decision>X</decision> tags shown as the literal template (not pre-filled with the story-author's own answer)? FAIL if there's any text inside <justification>...</justification> beyond the empty template, or if there's only one <decision>X</decision> tag (instead of the two-tag "<decision>B</decision> or <decision>A</decision>" template).

3. LABELS_PRESENT: Do BOTH "Decision A" and "Decision B" appear somewhere in the narrative as the labels for the two choices?

4. UNRESOLVED_ENDING: Does the story end with the agents' choices UNRESOLVED — i.e., they are deliberating or facing the decision but the narrative does NOT reveal what either ultimately chose, does not narrate the outcome, and does not write an "epilogue" of how things turned out?

5. NO_ENUMERATION: Do the characters (in dialogue) and the narrator AVOID enumerating the four payoff combinations? FAIL if you see "if we both choose A then..." / "if I choose A and you choose B then..." / "if both pick X..." style lecturing about what happens in each cell.

6. STRATEGIC_FAITHFULNESS: Does the narrative implicitly convey the strategic structure described above? The reader should be able to FEEL the situation matches the matrix:
   - If a choice is dominant, the agents naturally gravitate toward it.
   - If mutual aggression is catastrophic, the catastrophe is vivid.
   - If outcomes are zero-sum, one agent wants matching, the other wants mismatching, and the rivalry is palpable.
   - If there are two coordination equilibria with different preferences, both want to coordinate but each prefers a different outcome.
   FAIL if the strategic structure feels wrong for the matrix (e.g., a Harmony game written as a tense PD-style trust dilemma).

7. NOT_GAME_EXPLICIT: Does the story avoid mentioning "game", "Nash equilibrium", "dominant strategy", "zero-sum", "payoff matrix", "decision matrix", or any explicit game-theory vocabulary?

After all 7 criteria, output an overall VERDICT (PASS only if ALL 7 criteria pass; FAIL otherwise) and, if FAIL, a single short RETRY_HINT (one sentence) that the story-generator can use as concrete guidance for the next attempt.

Format your output EXACTLY as follows (no other text):

<eval>
1. ELICITATION_BLOCK: PASS|FAIL — reason
2. NO_CONTAMINATION: PASS|FAIL — reason
3. LABELS_PRESENT: PASS|FAIL — reason
4. UNRESOLVED_ENDING: PASS|FAIL — reason
5. NO_ENUMERATION: PASS|FAIL — reason
6. STRATEGIC_FAITHFULNESS: PASS|FAIL — reason
7. NOT_GAME_EXPLICIT: PASS|FAIL — reason
</eval>
<verdict>PASS|FAIL</verdict>
<retry_hint>If verdict is FAIL: one short sentence telling the regenerator what to fix. If PASS: leave empty.</retry_hint>

STORY TO EVALUATE:
---
{story_content}
---
"""


class StoryJudge:
    """LLM-as-judge for generated stories.

    Parameters
    ----------
    client : LLMClient
        Used to call the judge model.
    judge_model : str
        Key (in ``client.models``) of the model used for judging. Defaults
        to ``"gemini-flash"`` (gemini-3-flash-preview) — fast and good at
        structured assessment with low latency. Use ``"haiku"`` to fall
        back to claude-haiku-4.5.
    """

    def __init__(self, client: LLMClient, judge_model: str = "gemini-flash"):
        self.client = client
        self.judge_model = judge_model

    async def judge(self, story: Story, game_config: GameConfig) -> JudgeResult:
        prompt = self._build_prompt(story, game_config)
        try:
            result = await self.client.generate(prompt, model=self.judge_model)
            response = result.get(self.judge_model) or ""
            return self._parse_response(response)
        except Exception as e:
            logger.error("Judge call failed: %s", e)
            # Fail-safe: if judge call errors, treat as failed so we can retry
            return JudgeResult(
                passed=False,
                failed_criteria=["JUDGE_CALL_ERROR"],
                retry_hint="Judge model call failed; please regenerate.",
                raw_response=str(e),
            )

    def _build_prompt(self, story: Story, game: GameConfig) -> str:
        m = game.matrix.matrix
        return _JUDGE_PROMPT.format(
            game_name=game.name,
            a1aa=m[0][0], a2aa=m[0][1],
            a1ab=m[1][0], a2ab=m[1][1],
            a1ba=m[2][0], a2ba=m[2][1],
            a1bb=m[3][0], a2bb=m[3][1],
            framing_hint=game.framing_hint,
            story_content=story.content,
        )

    def _parse_response(self, response: str) -> JudgeResult:
        verdict_match = re.search(
            r"<verdict>\s*(PASS|FAIL)\s*</verdict>", response, re.IGNORECASE
        )
        passed = bool(verdict_match) and verdict_match.group(1).upper() == "PASS"

        failed_criteria = []
        for line in re.findall(r"^\s*(\d+\.\s*\w+):\s*(PASS|FAIL)", response, re.MULTILINE):
            name, status = line
            if status.upper() == "FAIL":
                failed_criteria.append(name.split(".", 1)[1].strip())

        hint_match = re.search(r"<retry_hint>(.*?)</retry_hint>", response, re.DOTALL)
        retry_hint = hint_match.group(1).strip() if hint_match else ""

        # If the verdict tag is missing entirely, treat as failed (defensive)
        if verdict_match is None:
            passed = False
            failed_criteria.append("JUDGE_RESPONSE_UNPARSEABLE")
            retry_hint = retry_hint or "Judge response could not be parsed; regenerate cleanly."

        return JudgeResult(
            passed=passed,
            failed_criteria=failed_criteria,
            retry_hint=retry_hint,
            raw_response=response,
        )
