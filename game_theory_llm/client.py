# game_theory_llm/client.py
"""LLM API client with rate limiting and retry logic.

All model calls are routed through OpenRouter's unified OpenAI-compatible API.
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ._logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    provider: str  # "openrouter"
    model_id: str  # OpenRouter model ID (e.g. "meta-llama/llama-3.3-70b-instruct")
    family: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    # Reasoning effort hint forwarded to OpenRouter's `reasoning.effort` field.
    # Applies only to reasoning-capable models (GPT-5.x, o-series, Opus,
    # Gemini Pro reasoning, Qwen *-thinking variants); ignored by the rest.
    # Set to None to omit the param entirely.
    reasoning_effort: Optional[str] = "medium"


# ---------------------------------------------------------------------------
# Model registry — all models via OpenRouter
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, ModelConfig]] = {

    # ── Gemma (Google, open-weight) ──────────────────────────────────────
    # Added 2026-07-24 for the AAAI-27 revision: NeurIPS reviewers asked for
    # the steering-section models in the behavioral battery. 26B-A4B is one
    # of the paper's steering models; 31B is the hosted family flagship.
    # (Gemma 4 E4B-it — the primary steering model — is not hosted on
    # OpenRouter; it is evaluated via the Modal "safety" workers instead.)
    "gemma": {
        "gemma-4-26b-a4b-it": ModelConfig("gemma-4-26b-a4b-it", "openrouter",
                                          "google/gemma-4-26b-a4b-it",
                                          family="gemma", reasoning_effort=None),
        "gemma-4-31b-it":     ModelConfig("gemma-4-31b-it",     "openrouter",
                                          "google/gemma-4-31b-it",
                                          family="gemma", reasoning_effort=None),
        "gemma-3n-e4b-it":    ModelConfig("gemma-3n-e4b-it",    "openrouter",
                                          "google/gemma-3n-e4b-it",
                                          family="gemma", reasoning_effort=None),
    },

    # ── Grok (xAI) ───────────────────────────────────────────────────────
    "grok": {
        # Grok 3
        "grok-3":           ModelConfig("grok-3",           "openrouter", "x-ai/grok-3",                family="grok"),
        "grok-3-beta":      ModelConfig("grok-3-beta",      "openrouter", "x-ai/grok-3-beta",           family="grok"),
        "grok-3-mini":      ModelConfig("grok-3-mini",      "openrouter", "x-ai/grok-3-mini",           family="grok"),
        "grok-3-mini-beta": ModelConfig("grok-3-mini-beta", "openrouter", "x-ai/grok-3-mini-beta",      family="grok"),
        # Grok 4
        "grok-4":           ModelConfig("grok-4",           "openrouter", "x-ai/grok-4",                family="grok"),
        "grok-4-fast":      ModelConfig("grok-4-fast",      "openrouter", "x-ai/grok-4-fast",           family="grok"),
        "grok-4.1-fast":    ModelConfig("grok-4.1-fast",    "openrouter", "x-ai/grok-4.1-fast",         family="grok"),
        "grok-4.20":        ModelConfig("grok-4.20",        "openrouter", "x-ai/grok-4.20",             family="grok"),
        "grok-4.20-multi":  ModelConfig("grok-4.20-multi",  "openrouter", "x-ai/grok-4.20-multi-agent", family="grok"),
        "grok-4.3":         ModelConfig("grok-4.3",         "openrouter", "x-ai/grok-4.3",              family="grok"),
        # Grok Code
        "grok-code-fast-1": ModelConfig("grok-code-fast-1", "openrouter", "x-ai/grok-code-fast-1",      family="grok"),
    },

    # ── Qwen (Alibaba) ──────────────────────────────────────────────────
    "qwen": {
        # Qwen 2.5
        "qwen-2.5-7b":             ModelConfig("qwen-2.5-7b",             "openrouter", "qwen/qwen-2.5-7b-instruct",         family="qwen"),
        "qwen-2.5-72b":            ModelConfig("qwen-2.5-72b",            "openrouter", "qwen/qwen-2.5-72b-instruct",        family="qwen"),
        "qwen-2.5-coder-32b":      ModelConfig("qwen-2.5-coder-32b",      "openrouter", "qwen/qwen-2.5-coder-32b-instruct",  family="qwen"),
        "qwen-2.5-vl-72b":         ModelConfig("qwen-2.5-vl-72b",         "openrouter", "qwen/qwen2.5-vl-72b-instruct",      family="qwen"),
        # Qwen proprietary
        "qwen-max":                 ModelConfig("qwen-max",                 "openrouter", "qwen/qwen-max",                     family="qwen"),
        "qwen-plus":                ModelConfig("qwen-plus",                "openrouter", "qwen/qwen-plus",                    family="qwen"),
        "qwen-plus-2507":           ModelConfig("qwen-plus-2507",           "openrouter", "qwen/qwen-plus-2025-07-28",         family="qwen"),
        "qwen-plus-2507-thinking":  ModelConfig("qwen-plus-2507-thinking",  "openrouter", "qwen/qwen-plus-2025-07-28:thinking",family="qwen"),
        "qwen-turbo":               ModelConfig("qwen-turbo",               "openrouter", "qwen/qwen-turbo",                   family="qwen"),
        "qwen-vl-max":              ModelConfig("qwen-vl-max",              "openrouter", "qwen/qwen-vl-max",                  family="qwen"),
        "qwen-vl-plus":             ModelConfig("qwen-vl-plus",             "openrouter", "qwen/qwen-vl-plus",                 family="qwen"),
        # Qwen 3 dense
        "qwen-3-8b":               ModelConfig("qwen-3-8b",               "openrouter", "qwen/qwen3-8b",                      family="qwen"),
        "qwen-3-14b":              ModelConfig("qwen-3-14b",              "openrouter", "qwen/qwen3-14b",                     family="qwen"),
        "qwen-3-32b":              ModelConfig("qwen-3-32b",              "openrouter", "qwen/qwen3-32b",                     family="qwen"),
        # Qwen 3 MoE
        "qwen-3-30b-a3b":          ModelConfig("qwen-3-30b-a3b",          "openrouter", "qwen/qwen3-30b-a3b",                 family="qwen"),
        "qwen-3-30b-a3b-2507":     ModelConfig("qwen-3-30b-a3b-2507",     "openrouter", "qwen/qwen3-30b-a3b-instruct-2507",   family="qwen"),
        "qwen-3-30b-a3b-thinking": ModelConfig("qwen-3-30b-a3b-thinking", "openrouter", "qwen/qwen3-30b-a3b-thinking-2507",   family="qwen"),
        "qwen-3-235b-a22b":        ModelConfig("qwen-3-235b-a22b",        "openrouter", "qwen/qwen3-235b-a22b",               family="qwen"),
        "qwen-3-235b-a22b-2507":   ModelConfig("qwen-3-235b-a22b-2507",   "openrouter", "qwen/qwen3-235b-a22b-2507",          family="qwen"),
        "qwen-3-235b-thinking":    ModelConfig("qwen-3-235b-thinking",    "openrouter", "qwen/qwen3-235b-a22b-thinking-2507", family="qwen"),
        # Qwen 3 Max & Next
        "qwen-3-max":              ModelConfig("qwen-3-max",              "openrouter", "qwen/qwen3-max",                     family="qwen"),
        "qwen-3-max-thinking":     ModelConfig("qwen-3-max-thinking",     "openrouter", "qwen/qwen3-max-thinking",            family="qwen"),
        "qwen-3-next-80b":         ModelConfig("qwen-3-next-80b",         "openrouter", "qwen/qwen3-next-80b-a3b-instruct",   family="qwen"),
        "qwen-3-next-80b-thinking":ModelConfig("qwen-3-next-80b-thinking","openrouter", "qwen/qwen3-next-80b-a3b-thinking",   family="qwen"),
        # Qwen 3 Coder
        "qwen-3-coder":            ModelConfig("qwen-3-coder",            "openrouter", "qwen/qwen3-coder",                   family="qwen"),
        "qwen-3-coder-30b":        ModelConfig("qwen-3-coder-30b",        "openrouter", "qwen/qwen3-coder-30b-a3b-instruct",  family="qwen"),
        "qwen-3-coder-flash":      ModelConfig("qwen-3-coder-flash",      "openrouter", "qwen/qwen3-coder-flash",             family="qwen"),
        "qwen-3-coder-next":       ModelConfig("qwen-3-coder-next",       "openrouter", "qwen/qwen3-coder-next",              family="qwen"),
        "qwen-3-coder-plus":       ModelConfig("qwen-3-coder-plus",       "openrouter", "qwen/qwen3-coder-plus",              family="qwen"),
        # Qwen 3 VL
        "qwen-3-vl-8b":            ModelConfig("qwen-3-vl-8b",            "openrouter", "qwen/qwen3-vl-8b-instruct",          family="qwen"),
        "qwen-3-vl-8b-thinking":   ModelConfig("qwen-3-vl-8b-thinking",   "openrouter", "qwen/qwen3-vl-8b-thinking",          family="qwen"),
        "qwen-3-vl-30b-a3b":       ModelConfig("qwen-3-vl-30b-a3b",       "openrouter", "qwen/qwen3-vl-30b-a3b-instruct",     family="qwen"),
        "qwen-3-vl-30b-thinking":  ModelConfig("qwen-3-vl-30b-thinking",  "openrouter", "qwen/qwen3-vl-30b-a3b-thinking",     family="qwen"),
        "qwen-3-vl-32b":           ModelConfig("qwen-3-vl-32b",           "openrouter", "qwen/qwen3-vl-32b-instruct",         family="qwen"),
        "qwen-3-vl-235b-a22b":     ModelConfig("qwen-3-vl-235b-a22b",     "openrouter", "qwen/qwen3-vl-235b-a22b-instruct",   family="qwen"),
        "qwen-3-vl-235b-thinking": ModelConfig("qwen-3-vl-235b-thinking", "openrouter", "qwen/qwen3-vl-235b-a22b-thinking",   family="qwen"),
        # Qwen 3.5
        "qwen-3.5-9b":             ModelConfig("qwen-3.5-9b",             "openrouter", "qwen/qwen3.5-9b",                    family="qwen"),
        "qwen-3.5-27b":            ModelConfig("qwen-3.5-27b",            "openrouter", "qwen/qwen3.5-27b",                   family="qwen"),
        "qwen-3.5-35b-a3b":        ModelConfig("qwen-3.5-35b-a3b",        "openrouter", "qwen/qwen3.5-35b-a3b",              family="qwen"),
        "qwen-3.5-122b-a10b":      ModelConfig("qwen-3.5-122b-a10b",      "openrouter", "qwen/qwen3.5-122b-a10b",            family="qwen"),
        "qwen-3.5-397b-a17b":      ModelConfig("qwen-3.5-397b-a17b",      "openrouter", "qwen/qwen3.5-397b-a17b",            family="qwen"),
        "qwen-3.5-flash":          ModelConfig("qwen-3.5-flash",          "openrouter", "qwen/qwen3.5-flash-02-23",           family="qwen"),
        "qwen-3.5-plus":           ModelConfig("qwen-3.5-plus",           "openrouter", "qwen/qwen3.5-plus-02-15",            family="qwen"),
        # Qwen 3.6
        "qwen-3.6-plus":           ModelConfig("qwen-3.6-plus",           "openrouter", "qwen/qwen3.6-plus",                  family="qwen"),
        # QwQ reasoning
        "qwq-32b":                 ModelConfig("qwq-32b",                 "openrouter", "qwen/qwq-32b",                       family="qwen"),
    },

    # ── DeepSeek ─────────────────────────────────────────────────────────
    "deepseek": {
        # V3
        "deepseek-v3":            ModelConfig("deepseek-v3",            "openrouter", "deepseek/deepseek-chat",                family="deepseek"),
        "deepseek-v3-0324":       ModelConfig("deepseek-v3-0324",       "openrouter", "deepseek/deepseek-chat-v3-0324",        family="deepseek"),
        "deepseek-v3.1":          ModelConfig("deepseek-v3.1",          "openrouter", "deepseek/deepseek-chat-v3.1",           family="deepseek"),
        "deepseek-v3.1-terminus": ModelConfig("deepseek-v3.1-terminus", "openrouter", "deepseek/deepseek-v3.1-terminus",       family="deepseek"),
        "deepseek-v3.2":          ModelConfig("deepseek-v3.2",          "openrouter", "deepseek/deepseek-v3.2",                family="deepseek"),
        "deepseek-v3.2-exp":      ModelConfig("deepseek-v3.2-exp",      "openrouter", "deepseek/deepseek-v3.2-exp",            family="deepseek"),
        "deepseek-v3.2-speciale": ModelConfig("deepseek-v3.2-speciale", "openrouter", "deepseek/deepseek-v3.2-speciale",       family="deepseek"),
        # V4
        "deepseek-v4-pro":        ModelConfig("deepseek-v4-pro",        "openrouter", "deepseek/deepseek-v4-pro",              family="deepseek", max_tokens=8000),
        "deepseek-v4-flash":      ModelConfig("deepseek-v4-flash",      "openrouter", "deepseek/deepseek-v4-flash",            family="deepseek"),
        # R1
        "deepseek-r1":            ModelConfig("deepseek-r1",            "openrouter", "deepseek/deepseek-r1",                  family="deepseek"),
        "deepseek-r1-0528":       ModelConfig("deepseek-r1-0528",       "openrouter", "deepseek/deepseek-r1-0528",             family="deepseek"),
        # R1 distills
        "deepseek-r1-llama-70b":  ModelConfig("deepseek-r1-llama-70b",  "openrouter", "deepseek/deepseek-r1-distill-llama-70b",family="deepseek"),
        "deepseek-r1-qwen-32b":   ModelConfig("deepseek-r1-qwen-32b",  "openrouter", "deepseek/deepseek-r1-distill-qwen-32b", family="deepseek"),
    },

    # ── Claude (Anthropic) ───────────────────────────────────────────────
    "claude": {
        # Claude 3
        "claude-3-haiku":          ModelConfig("claude-3-haiku",          "openrouter", "anthropic/claude-3-haiku",             family="claude"),
        # Claude 3.5 / 3.7
        "claude-3.5-haiku":        ModelConfig("claude-3.5-haiku",        "openrouter", "anthropic/claude-3.5-haiku",           family="claude"),
        "claude-3.7-sonnet":       ModelConfig("claude-3.7-sonnet",       "openrouter", "anthropic/claude-3.7-sonnet",          family="claude"),
        "claude-3.7-sonnet-think": ModelConfig("claude-3.7-sonnet-think", "openrouter", "anthropic/claude-3.7-sonnet:thinking", family="claude"),
        # Claude 4
        "claude-sonnet-4":         ModelConfig("claude-sonnet-4",         "openrouter", "anthropic/claude-sonnet-4",            family="claude"),
        "claude-opus-4":           ModelConfig("claude-opus-4",           "openrouter", "anthropic/claude-opus-4",              family="claude"),
        "claude-opus-4.1":         ModelConfig("claude-opus-4.1",         "openrouter", "anthropic/claude-opus-4.1",            family="claude"),
        # Claude 4.5
        "claude-haiku-4.5":        ModelConfig("claude-haiku-4.5",        "openrouter", "anthropic/claude-haiku-4.5",           family="claude"),
        "claude-sonnet-4.5":       ModelConfig("claude-sonnet-4.5",       "openrouter", "anthropic/claude-sonnet-4.5",          family="claude"),
        "claude-opus-4.5":         ModelConfig("claude-opus-4.5",         "openrouter", "anthropic/claude-opus-4.5",            family="claude"),
        # Claude 4.6
        "claude-sonnet-4.6":       ModelConfig("claude-sonnet-4.6",       "openrouter", "anthropic/claude-sonnet-4.6",          family="claude"),
        "claude-opus-4.6":         ModelConfig("claude-opus-4.6",         "openrouter", "anthropic/claude-opus-4.6",            family="claude"),
        "claude-opus-4.6-fast":    ModelConfig("claude-opus-4.6-fast",    "openrouter", "anthropic/claude-opus-4.6-fast",       family="claude"),
        # Claude 4.7
        "claude-opus-4.7":         ModelConfig("claude-opus-4.7",         "openrouter", "anthropic/claude-opus-4.7",            family="claude"),
    },

    # ── GPT / OpenAI ─────────────────────────────────────────────────────
    "gpt": {
        # GPT-3.5
        "gpt-3.5-turbo":       ModelConfig("gpt-3.5-turbo",       "openrouter", "openai/gpt-3.5-turbo",              family="gpt"),
        "gpt-3.5-turbo-0613":  ModelConfig("gpt-3.5-turbo-0613",  "openrouter", "openai/gpt-3.5-turbo-0613",         family="gpt"),
        "gpt-3.5-turbo-16k":   ModelConfig("gpt-3.5-turbo-16k",   "openrouter", "openai/gpt-3.5-turbo-16k",          family="gpt"),
        # GPT-4 base
        "gpt-4":               ModelConfig("gpt-4",               "openrouter", "openai/gpt-4",                      family="gpt"),
        "gpt-4-turbo":         ModelConfig("gpt-4-turbo",         "openrouter", "openai/gpt-4-turbo",                family="gpt"),
        # GPT-4.1
        "gpt-4.1":             ModelConfig("gpt-4.1",             "openrouter", "openai/gpt-4.1",                    family="gpt"),
        "gpt-4.1-mini":        ModelConfig("gpt-4.1-mini",        "openrouter", "openai/gpt-4.1-mini",               family="gpt"),
        "gpt-4.1-nano":        ModelConfig("gpt-4.1-nano",        "openrouter", "openai/gpt-4.1-nano",               family="gpt"),
        # GPT-4o
        "gpt-4o":              ModelConfig("gpt-4o",              "openrouter", "openai/gpt-4o",                     family="gpt"),
        "gpt-4o-may24":        ModelConfig("gpt-4o-may24",        "openrouter", "openai/gpt-4o-2024-05-13",          family="gpt"),
        "gpt-4o-aug24":        ModelConfig("gpt-4o-aug24",        "openrouter", "openai/gpt-4o-2024-08-06",          family="gpt"),
        "gpt-4o-nov24":        ModelConfig("gpt-4o-nov24",        "openrouter", "openai/gpt-4o-2024-11-20",          family="gpt"),
        "gpt-4o-mini":         ModelConfig("gpt-4o-mini",         "openrouter", "openai/gpt-4o-mini",                family="gpt"),
        "gpt-4o-mini-jul24":   ModelConfig("gpt-4o-mini-jul24",   "openrouter", "openai/gpt-4o-mini-2024-07-18",     family="gpt"),
        "gpt-4o-mini-search":  ModelConfig("gpt-4o-mini-search",  "openrouter", "openai/gpt-4o-mini-search-preview", family="gpt"),
        "gpt-4o-search":       ModelConfig("gpt-4o-search",       "openrouter", "openai/gpt-4o-search-preview",      family="gpt"),
        # GPT-5
        "gpt-5":               ModelConfig("gpt-5",               "openrouter", "openai/gpt-5",                      family="gpt"),
        "gpt-5-chat":          ModelConfig("gpt-5-chat",          "openrouter", "openai/gpt-5-chat",                 family="gpt"),
        "gpt-5-codex":         ModelConfig("gpt-5-codex",         "openrouter", "openai/gpt-5-codex",                family="gpt"),
        "gpt-5-mini":          ModelConfig("gpt-5-mini",          "openrouter", "openai/gpt-5-mini",                 family="gpt"),
        "gpt-5-nano":          ModelConfig("gpt-5-nano",          "openrouter", "openai/gpt-5-nano",                 family="gpt"),
        "gpt-5-pro":           ModelConfig("gpt-5-pro",           "openrouter", "openai/gpt-5-pro",                  family="gpt"),
        # GPT-5.1
        "gpt-5.1":             ModelConfig("gpt-5.1",             "openrouter", "openai/gpt-5.1",                    family="gpt"),
        "gpt-5.1-chat":        ModelConfig("gpt-5.1-chat",        "openrouter", "openai/gpt-5.1-chat",               family="gpt"),
        "gpt-5.1-codex":       ModelConfig("gpt-5.1-codex",       "openrouter", "openai/gpt-5.1-codex",              family="gpt"),
        "gpt-5.1-codex-max":   ModelConfig("gpt-5.1-codex-max",   "openrouter", "openai/gpt-5.1-codex-max",          family="gpt"),
        "gpt-5.1-codex-mini":  ModelConfig("gpt-5.1-codex-mini",  "openrouter", "openai/gpt-5.1-codex-mini",         family="gpt"),
        # GPT-5.2
        "gpt-5.2":             ModelConfig("gpt-5.2",             "openrouter", "openai/gpt-5.2",                    family="gpt"),
        "gpt-5.2-chat":        ModelConfig("gpt-5.2-chat",        "openrouter", "openai/gpt-5.2-chat",               family="gpt"),
        "gpt-5.2-codex":       ModelConfig("gpt-5.2-codex",       "openrouter", "openai/gpt-5.2-codex",              family="gpt"),
        "gpt-5.2-pro":         ModelConfig("gpt-5.2-pro",         "openrouter", "openai/gpt-5.2-pro",                family="gpt"),
        # GPT-5.3
        "gpt-5.3-chat":        ModelConfig("gpt-5.3-chat",        "openrouter", "openai/gpt-5.3-chat",               family="gpt"),
        "gpt-5.3-codex":       ModelConfig("gpt-5.3-codex",       "openrouter", "openai/gpt-5.3-codex",              family="gpt"),
        # GPT-5.4
        "gpt-5.4":             ModelConfig("gpt-5.4",             "openrouter", "openai/gpt-5.4",                    family="gpt"),
        "gpt-5.4-mini":        ModelConfig("gpt-5.4-mini",        "openrouter", "openai/gpt-5.4-mini",               family="gpt"),
        "gpt-5.4-nano":        ModelConfig("gpt-5.4-nano",        "openrouter", "openai/gpt-5.4-nano",               family="gpt"),
        "gpt-5.4-pro":         ModelConfig("gpt-5.4-pro",         "openrouter", "openai/gpt-5.4-pro",                family="gpt"),
        "gpt-5.5":             ModelConfig("gpt-5.5",             "openrouter", "openai/gpt-5.5",                    family="gpt"),
        "gpt-5.5-pro":         ModelConfig("gpt-5.5-pro",         "openrouter", "openai/gpt-5.5-pro",                family="gpt"),
        # GPT open-source
        "gpt-oss-120b":        ModelConfig("gpt-oss-120b",        "openrouter", "openai/gpt-oss-120b",               family="gpt"),
        "gpt-oss-20b":         ModelConfig("gpt-oss-20b",         "openrouter", "openai/gpt-oss-20b",                family="gpt"),
        # o-series reasoning
        "o1":                  ModelConfig("o1",                  "openrouter", "openai/o1",                 family="gpt", max_tokens=16000),
        "o1-pro":              ModelConfig("o1-pro",              "openrouter", "openai/o1-pro",             family="gpt", max_tokens=16000),
        "o3":                  ModelConfig("o3",                  "openrouter", "openai/o3",                 family="gpt", max_tokens=16000),
        "o3-mini":             ModelConfig("o3-mini",             "openrouter", "openai/o3-mini",            family="gpt"),
        "o3-mini-high":        ModelConfig("o3-mini-high",        "openrouter", "openai/o3-mini-high",       family="gpt"),
        "o3-pro":              ModelConfig("o3-pro",              "openrouter", "openai/o3-pro",             family="gpt", max_tokens=16000),
        "o4-mini":             ModelConfig("o4-mini",             "openrouter", "openai/o4-mini",            family="gpt"),
        "o4-mini-high":        ModelConfig("o4-mini-high",        "openrouter", "openai/o4-mini-high",       family="gpt"),
    },

    # ── Gemini (Google) ──────────────────────────────────────────────────
    "gemini": {
        # Gemini 2.0  (1.0 / 1.5 are no longer on OpenRouter)
        "gemini-2.0-flash":           ModelConfig("gemini-2.0-flash",           "openrouter", "google/gemini-2.0-flash-001",                family="gemini"),
        "gemini-2.0-flash-lite":      ModelConfig("gemini-2.0-flash-lite",      "openrouter", "google/gemini-2.0-flash-lite-001",           family="gemini"),
        # Gemini 2.5
        "gemini-2.5-flash-lite":      ModelConfig("gemini-2.5-flash-lite",      "openrouter", "google/gemini-2.5-flash-lite",               family="gemini"),
        "gemini-2.5-flash-lite-0925": ModelConfig("gemini-2.5-flash-lite-0925", "openrouter", "google/gemini-2.5-flash-lite-preview-09-2025",family="gemini"),
        "gemini-2.5-flash":           ModelConfig("gemini-2.5-flash",           "openrouter", "google/gemini-2.5-flash",                    family="gemini"),
        "gemini-2.5-pro":             ModelConfig("gemini-2.5-pro",             "openrouter", "google/gemini-2.5-pro",                      family="gemini"),
        "gemini-2.5-pro-preview":     ModelConfig("gemini-2.5-pro-preview",     "openrouter", "google/gemini-2.5-pro-preview",              family="gemini"),
        "gemini-2.5-pro-0506":        ModelConfig("gemini-2.5-pro-0506",        "openrouter", "google/gemini-2.5-pro-preview-05-06",        family="gemini"),
        # Gemini 3
        "gemini-3-flash":             ModelConfig("gemini-3-flash",             "openrouter", "google/gemini-3-flash-preview",              family="gemini", max_tokens=16000),
        "gemini-3.1-flash-lite":      ModelConfig("gemini-3.1-flash-lite",      "openrouter", "google/gemini-3.1-flash-lite-preview",       family="gemini"),
        "gemini-3.1-pro":             ModelConfig("gemini-3.1-pro",             "openrouter", "google/gemini-3.1-pro-preview",              family="gemini"),
    },
}


DEFAULT_MODELS: Dict[str, ModelConfig] = {
    "opus":         MODEL_REGISTRY["claude"]["claude-opus-4.7"],
    "gpt-5.4-mini": MODEL_REGISTRY["gpt"]["gpt-5.4-mini"],
    "ds-v4-pro":    MODEL_REGISTRY["deepseek"]["deepseek-v4-pro"],
    "haiku":        MODEL_REGISTRY["claude"]["claude-haiku-4.5"],
    "gemini-flash": MODEL_REGISTRY["gemini"]["gemini-3-flash"],
    "deepseek":     MODEL_REGISTRY["deepseek"]["deepseek-v3"],
    "claude":       MODEL_REGISTRY["claude"]["claude-sonnet-4.6"],
    "gpt4":         MODEL_REGISTRY["gpt"]["gpt-4.1"],
}


def get_models(family: str) -> Dict[str, ModelConfig]:
    """Return all ModelConfigs for a given family."""
    return dict(MODEL_REGISTRY.get(family, {}))


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-API rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 1400,
        tokens_per_minute: int = 200_000,
        min_request_interval: float = 1 / 20,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.min_request_interval = min_request_interval
        self.request_timestamps: List[float] = []
        self.token_usage: List[Tuple[float, int]] = []
        self.last_request_time: float = 0

    async def enforce(self, tokens_to_use: int = 0) -> Tuple[bool, float]:
        """Return ``(can_proceed, wait_seconds)``."""
        current_time = time.time()
        minute_ago = current_time - 60

        # Minimum interval
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        # Prune old entries
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        self.token_usage = [(ts, tok) for ts, tok in self.token_usage if ts > minute_ago]

        # Request rate check
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = max(self.request_timestamps[0] - minute_ago + 0.1, 0.1)
            return False, sleep_time

        # Token rate check
        current_token_usage = sum(tok for _, tok in self.token_usage)
        if current_token_usage + tokens_to_use > self.tokens_per_minute:
            sleep_time = max(self.token_usage[0][0] - minute_ago + 0.1, 0.1)
            return False, sleep_time

        # Record
        self.request_timestamps.append(current_time)
        if tokens_to_use > 0:
            self.token_usage.append((current_time, tokens_to_use))
        self.last_request_time = current_time
        return True, 0

    def update_token_usage(self, actual_tokens: int) -> None:
        """Replace the last estimated token entry with the actual count."""
        if self.token_usage:
            self.token_usage[-1] = (self.token_usage[-1][0], actual_tokens)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified async client routing all models through OpenRouter.

    Parameters
    ----------
    models : dict[str, ModelConfig] | None
        Model configs keyed by short name.  Defaults to ``DEFAULT_MODELS``.
    api_key : str | None
        OpenRouter API key.  Falls back to ``OPENROUTER_API_KEY`` env var.
    zero_data_retention : bool
        When True (default), every request includes the OpenRouter provider
        preference ``data_collection: "deny"``, restricting routing to
        providers that do NOT store request data. Set False only if you need
        access to providers that don't support ZDR.
    """

    def __init__(
        self,
        models: Optional[Dict[str, ModelConfig]] = None,
        api_key: Optional[str] = None,
        zero_data_retention: bool = True,
    ):
        self.models = models or dict(DEFAULT_MODELS)
        self.zero_data_retention = zero_data_retention

        from openai import AsyncOpenAI
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter API key required via api_key param or OPENROUTER_API_KEY env var"
            )
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        self._rate_limiter = RateLimiter()

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _should_retry(error: Exception) -> bool:
        msg = str(error).lower()
        # Catch transient backend failures and malformed/empty responses that
        # show up under high concurrency: OpenRouter occasionally returns a
        # non-JSON body or an empty response, which the OpenAI SDK surfaces
        # as a JSONDecodeError ("expecting value: ...").
        return any(
            kw in msg
            for kw in (
                "rate_limit", "429", "503", "502", "504",
                "service unavailable", "capacity", "timeout", "server error",
                "expecting value", "json", "connection", "remote disconnected",
                "incomplete read", "bad gateway", "overloaded",
            )
        )

    def _extra_body(self, config: Optional[ModelConfig] = None) -> Dict:
        body: Dict = {}
        if self.zero_data_retention:
            body["provider"] = {"data_collection": "deny"}
        if config is not None and config.reasoning_effort is not None:
            # OpenRouter normalises this across providers (OpenAI's
            # reasoning_effort, Anthropic's thinking budget, Gemini's
            # reasoning config, Qwen's thinking flag).
            body["reasoning"] = {"effort": config.reasoning_effort}
        return body

    async def _call(self, prompt: str, config: ModelConfig) -> str:
        can_proceed, wait = await self._rate_limiter.enforce(config.max_tokens)
        if not can_proceed:
            await asyncio.sleep(wait)
        response = await self._client.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            extra_body=self._extra_body(config),
        )
        return response.choices[0].message.content

    async def _call_messages(self, messages: List[Dict], config: ModelConfig) -> str:
        can_proceed, wait = await self._rate_limiter.enforce(config.max_tokens)
        if not can_proceed:
            await asyncio.sleep(wait)
        response = await self._client.chat.completions.create(
            model=config.model_id,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            extra_body=self._extra_body(config),
        )
        return response.choices[0].message.content

    # -- public API --------------------------------------------------------

    async def generate_messages(
        self,
        messages: List[Dict],
        model: str = "all",
    ) -> Dict[str, Optional[str]]:
        """Generate text from one or more LLMs using a full message list.

        Parameters
        ----------
        messages : list[dict]
            Chat messages (``[{"role": "user", "content": "..."}, ...]``).
        model : str
            ``"all"`` for every configured model, or a single model name.

        Returns
        -------
        dict[str, str | None]
            Mapping from model name to response text (or *None* on failure).
        """
        max_retries = 10
        base_wait = 3
        responses: Dict[str, Optional[str]] = {}

        targets = list(self.models) if model == "all" else [model]

        for name in targets:
            config = self.models[name]
            for attempt in range(max_retries):
                try:
                    responses[name] = await self._call_messages(messages, config)
                    break
                except Exception as e:
                    if self._should_retry(e) and attempt < max_retries - 1:
                        wait = (base_wait ** (attempt + 1)) + random.uniform(1, 5)
                        logger.warning(
                            "Retrying %s, waiting %.2fs (attempt %d/%d)",
                            name, wait, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error("Error with %s: %s", name, e)
                        responses[name] = None
                        break

        return responses

    async def generate(
        self,
        prompt: str,
        model: str = "all",
    ) -> Dict[str, Optional[str]]:
        """Generate text from one or more LLMs.

        Parameters
        ----------
        prompt : str
        model : str
            ``"all"`` for every configured model, or a single model name
            (e.g. ``"llama"``, ``"claude"``, ``"gpt4"``).

        Returns
        -------
        dict[str, str | None]
            Mapping from model name to response text (or *None* on failure).
        """
        max_retries = 10
        base_wait = 3
        responses: Dict[str, Optional[str]] = {}

        targets = list(self.models) if model == "all" else [model]

        for name in targets:
            config = self.models[name]
            for attempt in range(max_retries):
                try:
                    responses[name] = await self._call(prompt, config)
                    break
                except Exception as e:
                    if self._should_retry(e) and attempt < max_retries - 1:
                        wait = (base_wait ** (attempt + 1)) + random.uniform(1, 5)
                        logger.warning(
                            "Retrying %s, waiting %.2fs (attempt %d/%d)",
                            name, wait, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error("Error with %s: %s", name, e)
                        responses[name] = None
                        break

        return responses
