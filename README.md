# Contextual Game Theoretic LLM Evaluations

This repository contains code for evaluating Large Language Models (LLMs) using game theoretic approaches in contextual settings. The framework allows for analyzing LLM behavior and decision-making in strategic scenarios.

## Overview

The project implements various game theory concepts to evaluate how LLMs perform in strategic decision-making tasks. It provides tools for:

- Generating game scenarios
- Evaluating LLM responses in game-theoretic contexts
- Analyzing results across different models and parameters
- Conducting ablation studies

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Contextual-Game-Theoretic-LLM-Evaluations.git
cd Contextual-Game-Theoretic-LLM-Evaluations

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

See example scripts and notebooks for usage examples:

- `example.py`: Basic usage example
- `example_analysis.py`: Analysis example
- `game_theory_analysis.ipynb`: Jupyter notebook with analysis examples

## Structure

- `game_theory_llm/`: Main package
  - `api.py`: API interfaces for LLM interactions
  - `generator.py`: Game scenario generation
  - `analyzer.py`: Analysis tools
  - `models.py`: Model definitions
  - `config.py`: Configuration settings

## License

[MIT License](LICENSE) 