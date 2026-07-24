# Contextual Game Theoretic LLM Evaluations

Framework for evaluating LLM decision-making through game-theoretic vignettes. Generates contextual scenarios from payoff matrices, sends them to multiple LLMs (Llama, Claude, GPT-4), and analyses decision patterns with statistical tests and visualisations.

Based on the methodology from *Framing the Game: A Generative Approach to Contextual LLM Evaluation*.

## Reproducing experiments

See **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** for the full reproduction index: every paper result mapped to its corpus-build / run / analyze commands, the steering run-id registry, and which data is committed vs regenerable. It covers the behavioral sweep, mechanistic probing, PD cooperation steering across three Gemma variants, cross-benchmark moral transfer, the reasoning- and coding-regression ablations (MMLU, GPQA-Diamond, GSM8k, HumanEval), the trust vector, and the multi-agent steering-in-games experiments.

## Installation

```bash
git clone https://github.com/yourusername/Contextual-Game-Theoretic-LLM-Evaluations.git
cd Contextual-Game-Theoretic-LLM-Evaluations
pip install -e ".[dev]"
```

## Environment Variables

Create a `.env` file (or export directly):

```
TOGETHER_API_KEY=your-together-key
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

## Quick Start

```python
import asyncio
import logging
from dotenv import load_dotenv
from game_theory_llm import LLMClient, PayoffMatrix, StoryGenerator

load_dotenv()
logging.basicConfig(level=logging.INFO)

async def main():
    matrix = PayoffMatrix([(0,0), (100,-50), (-50,100), (75,75)])
    client = LLMClient()
    generator = StoryGenerator(client)
    stories = await generator.generate_stories(
        payoff_matrix=matrix,
        topic="international business",
        world_type="real_world",
        actor_type="enemies",
        n_stories=2,
    )
    for story in stories:
        print(story.content[:200], "\n")

asyncio.run(main())
```

See `examples/basic_generation.py` and `examples/full_analysis.py` for more complete usage.

## Project Structure

```
game_theory_llm/
    __init__.py             # Public API
    models.py               # PayoffMatrix, Story, AnalysisResult, BatchGenerationResult
    config.py               # Experiment configs with presets (full, small, holdout)
    client.py               # LLM API client with rate limiting
    decision_parser.py      # Decision extraction from LLM responses
    generator.py            # Story generation
    _logging.py             # Logger factory
    analysis/
        __init__.py
        base.py             # Core analysis (StoryAnalyzer)
        statistical.py      # Chi-square, Fisher, Cramer's V, etc. (StatisticalAnalyzer)
        visualization.py    # Plotting functions
examples/
    basic_generation.py     # Minimal generation example
    full_analysis.py        # Generation + analysis + visualisation
tests/
    conftest.py             # Shared fixtures
    test_models.py
    test_decision_parser.py
    test_config.py
    test_generator.py
    test_analysis.py
    test_client.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

[MIT License](LICENSE)
