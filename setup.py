# setup.py
from setuptools import setup, find_packages

setup(
    name="game_theory_llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "together",
        "pandas"
    ]
)