# game_theory_llm/client.py
"""LLM API client with rate limiting and retry logic."""

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
    provider: str  # "fireworks" | "anthropic" | "openai" | "google"
    model_id: str
    family: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, ModelConfig]] = {

    "llama": {
        # Llama 2
        "llama-2-7b":       ModelConfig("llama-2-7b",  "fireworks", "accounts/fireworks/models/llama-v2-7b-chat",          family="llama"),
        "llama-2-13b":      ModelConfig("llama-2-13b", "fireworks", "accounts/fireworks/models/llama-v2-13b-chat",         family="llama"),
        "llama-2-70b":      ModelConfig("llama-2-70b", "fireworks", "accounts/fireworks/models/llama-v2-70b-chat",         family="llama"),
        # Llama 3
        "llama-3-8b":       ModelConfig("llama-3-8b",  "fireworks", "accounts/fireworks/models/llama-v3-8b-instruct",      family="llama"),
        "llama-3-70b":      ModelConfig("llama-3-70b", "fireworks", "accounts/fireworks/models/llama-v3-70b-instruct",     family="llama"),
        # Llama 3.1
        "llama-3.1-8b":     ModelConfig("llama-3.1-8b",   "fireworks", "accounts/fireworks/models/llama-v3p1-8b-instruct",   family="llama"),
        "llama-3.1-70b":    ModelConfig("llama-3.1-70b",  "fireworks", "accounts/fireworks/models/llama-v3p1-70b-instruct",  family="llama"),
        "llama-3.1-405b":   ModelConfig("llama-3.1-405b", "fireworks", "accounts/fireworks/models/llama-v3p1-405b-instruct", family="llama"),
        # Llama 3.2
        "llama-3.2-1b":     ModelConfig("llama-3.2-1b",  "fireworks", "accounts/fireworks/models/llama-v3p2-1b-instruct",  family="llama"),
        "llama-3.2-3b":     ModelConfig("llama-3.2-3b",  "fireworks", "accounts/fireworks/models/llama-v3p2-3b-instruct",  family="llama"),
        "llama-3.2-11b":    ModelConfig("llama-3.2-11b", "fireworks", "accounts/fireworks/models/llama-v3p2-11b-vision-instruct", family="llama"),
        "llama-3.2-90b":    ModelConfig("llama-3.2-90b", "fireworks", "accounts/fireworks/models/llama-v3p2-90b-vision-instruct", family="llama"),
        # Llama 3.3
        "llama-3.3-70b":    ModelConfig("llama-3.3-70b", "fireworks", "accounts/fireworks/models/llama-v3p3-70b-instruct", family="llama"),
        # Llama 4
        "llama-4-scout":    ModelConfig("llama-4-scout",    "fireworks", "accounts/fireworks/models/llama4-scout-instruct-basic",    family="llama"),
        "llama-4-maverick": ModelConfig("llama-4-maverick", "fireworks", "accounts/fireworks/models/llama4-maverick-instruct-basic", family="llama"),
    },

    "qwen": {
        # Qwen 1.5
        "qwen-1.5-72b":           ModelConfig("qwen-1.5-72b",           "fireworks", "accounts/fireworks/models/qwen1p5-72b-chat",                  family="qwen"),
        # Qwen 2
        "qwen-2-7b":              ModelConfig("qwen-2-7b",              "fireworks", "accounts/fireworks/models/qwen2-7b-instruct",                  family="qwen"),
        "qwen-2-72b":             ModelConfig("qwen-2-72b",             "fireworks", "accounts/fireworks/models/qwen2-72b-instruct",                 family="qwen"),
        # Qwen 2 VL
        "qwen-2-vl-7b":           ModelConfig("qwen-2-vl-7b",           "fireworks", "accounts/fireworks/models/qwen2-vl-7b-instruct",               family="qwen"),
        "qwen-2-vl-72b":          ModelConfig("qwen-2-vl-72b",          "fireworks", "accounts/fireworks/models/qwen2-vl-72b-instruct",              family="qwen"),
        # Qwen 2.5
        "qwen-2.5-0.5b":          ModelConfig("qwen-2.5-0.5b",          "fireworks", "accounts/fireworks/models/qwen2p5-0p5b-instruct",              family="qwen"),
        "qwen-2.5-1.5b":          ModelConfig("qwen-2.5-1.5b",          "fireworks", "accounts/fireworks/models/qwen2p5-1p5b-instruct",              family="qwen"),
        "qwen-2.5-3b":            ModelConfig("qwen-2.5-3b",            "fireworks", "accounts/fireworks/models/qwen2p5-3b-instruct",                family="qwen"),
        "qwen-2.5-7b":            ModelConfig("qwen-2.5-7b",            "fireworks", "accounts/fireworks/models/qwen2p5-7b-instruct",                family="qwen"),
        "qwen-2.5-14b":           ModelConfig("qwen-2.5-14b",           "fireworks", "accounts/fireworks/models/qwen2p5-14b-instruct",               family="qwen"),
        "qwen-2.5-32b":           ModelConfig("qwen-2.5-32b",           "fireworks", "accounts/fireworks/models/qwen2p5-32b-instruct",               family="qwen"),
        "qwen-2.5-72b":           ModelConfig("qwen-2.5-72b",           "fireworks", "accounts/fireworks/models/qwen2p5-72b-instruct",               family="qwen"),
        # Qwen 2.5 Coder
        "qwen-2.5-coder-1.5b":    ModelConfig("qwen-2.5-coder-1.5b",    "fireworks", "accounts/fireworks/models/qwen2p5-coder-1p5b-instruct",        family="qwen"),
        "qwen-2.5-coder-3b":      ModelConfig("qwen-2.5-coder-3b",      "fireworks", "accounts/fireworks/models/qwen2p5-coder-3b-instruct",          family="qwen"),
        "qwen-2.5-coder-7b":      ModelConfig("qwen-2.5-coder-7b",      "fireworks", "accounts/fireworks/models/qwen2p5-coder-7b-instruct",          family="qwen"),
        "qwen-2.5-coder-14b":     ModelConfig("qwen-2.5-coder-14b",     "fireworks", "accounts/fireworks/models/qwen2p5-coder-14b-instruct",         family="qwen"),
        "qwen-2.5-coder-32b":     ModelConfig("qwen-2.5-coder-32b",     "fireworks", "accounts/fireworks/models/qwen2p5-coder-32b-instruct",         family="qwen"),
        # Qwen 2.5 VL
        "qwen-2.5-vl-3b":         ModelConfig("qwen-2.5-vl-3b",         "fireworks", "accounts/fireworks/models/qwen2p5-vl-3b-instruct",             family="qwen"),
        "qwen-2.5-vl-7b":         ModelConfig("qwen-2.5-vl-7b",         "fireworks", "accounts/fireworks/models/qwen2p5-vl-7b-instruct",             family="qwen"),
        "qwen-2.5-vl-32b":        ModelConfig("qwen-2.5-vl-32b",        "fireworks", "accounts/fireworks/models/qwen2p5-vl-32b-instruct",            family="qwen"),
        "qwen-2.5-vl-72b":        ModelConfig("qwen-2.5-vl-72b",        "fireworks", "accounts/fireworks/models/qwen2p5-vl-72b-instruct",            family="qwen"),
        # Qwen 3 dense
        "qwen-3-0.6b":            ModelConfig("qwen-3-0.6b",            "fireworks", "accounts/fireworks/models/qwen3-0p6b",                         family="qwen"),
        "qwen-3-1.7b":            ModelConfig("qwen-3-1.7b",            "fireworks", "accounts/fireworks/models/qwen3-1p7b",                         family="qwen"),
        "qwen-3-4b":              ModelConfig("qwen-3-4b",              "fireworks", "accounts/fireworks/models/qwen3-4b",                           family="qwen"),
        "qwen-3-8b":              ModelConfig("qwen-3-8b",              "fireworks", "accounts/fireworks/models/qwen3-8b",                           family="qwen"),
        "qwen-3-14b":             ModelConfig("qwen-3-14b",             "fireworks", "accounts/fireworks/models/qwen3-14b",                          family="qwen"),
        "qwen-3-32b":             ModelConfig("qwen-3-32b",             "fireworks", "accounts/fireworks/models/qwen3-32b",                          family="qwen"),
        # Qwen 3 MoE
        "qwen-3-30b-a3b":         ModelConfig("qwen-3-30b-a3b",         "fireworks", "accounts/fireworks/models/qwen3-30b-a3b",                      family="qwen"),
        "qwen-3-235b-a22b":       ModelConfig("qwen-3-235b-a22b",       "fireworks", "accounts/fireworks/models/qwen3-235b-a22b",                    family="qwen"),
        "qwen-3-235b-thinking":   ModelConfig("qwen-3-235b-thinking",   "fireworks", "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507",       family="qwen"),
        # Qwen 3 Coder
        "qwen-3-coder-30b":       ModelConfig("qwen-3-coder-30b",       "fireworks", "accounts/fireworks/models/qwen3-coder-30b-a3b-instruct",        family="qwen"),
        "qwen-3-coder-480b":      ModelConfig("qwen-3-coder-480b",      "fireworks", "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",      family="qwen"),
        "qwen-3-coder-next":      ModelConfig("qwen-3-coder-next",      "fireworks", "accounts/fireworks/models/qwen3-next-80b-a3b-instruct",         family="qwen"),
        # QwQ reasoning
        "qwq-32b-preview":        ModelConfig("qwq-32b-preview",        "fireworks", "accounts/fireworks/models/qwen-qwq-32b-preview",               family="qwen"),
        "qwq-32b":                ModelConfig("qwq-32b",                "fireworks", "accounts/fireworks/models/qwq-32b",                            family="qwen"),
    },

    "deepseek": {
        "deepseek-v3":           ModelConfig("deepseek-v3",          "fireworks", "accounts/fireworks/models/deepseek-v3",                        family="deepseek"),
        "deepseek-v3.1":         ModelConfig("deepseek-v3.1",        "fireworks", "accounts/fireworks/models/deepseek-v3p1",                      family="deepseek"),
        "deepseek-v3.2":         ModelConfig("deepseek-v3.2",        "fireworks", "accounts/fireworks/models/deepseek-v3p2",                      family="deepseek"),
        "deepseek-r1":           ModelConfig("deepseek-r1",          "fireworks", "accounts/fireworks/models/deepseek-r1",                        family="deepseek"),
        "deepseek-r1-0528":      ModelConfig("deepseek-r1-0528",     "fireworks", "accounts/fireworks/models/deepseek-r1-0528",                   family="deepseek"),
        "deepseek-r1-llama-70b": ModelConfig("deepseek-r1-llama-70b","fireworks", "accounts/fireworks/models/deepseek-r1-distill-llama-70b",      family="deepseek"),
        "deepseek-r1-qwen-32b":  ModelConfig("deepseek-r1-qwen-32b", "fireworks", "accounts/fireworks/models/deepseek-r1-distill-qwen-32b",       family="deepseek"),
    },

    "mistral": {
        "mistral-7b":      ModelConfig("mistral-7b",      "fireworks", "accounts/fireworks/models/mistral-7b-instruct-v3",          family="mistral"),
        "mixtral-8x7b":    ModelConfig("mixtral-8x7b",    "fireworks", "accounts/fireworks/models/mixtral-8x7b-instruct",           family="mistral"),
        "mixtral-8x22b":   ModelConfig("mixtral-8x22b",   "fireworks", "accounts/fireworks/models/mixtral-8x22b-instruct",          family="mistral"),
        "mistral-nemo":    ModelConfig("mistral-nemo",    "fireworks", "accounts/fireworks/models/mistral-nemo-instruct-2407",      family="mistral"),
        "mistral-small":   ModelConfig("mistral-small",   "fireworks", "accounts/fireworks/models/mistral-small-24b-instruct-2501", family="mistral"),
        "mistral-large-3": ModelConfig("mistral-large-3", "fireworks", "accounts/fireworks/models/mistral-large-3-fp8",            family="mistral"),
    },

    "gemma": {
        "gemma-7b":    ModelConfig("gemma-7b",    "fireworks", "accounts/fireworks/models/gemma-7b-it",    family="gemma"),
        "gemma-2-9b":  ModelConfig("gemma-2-9b",  "fireworks", "accounts/fireworks/models/gemma2-9b-it",  family="gemma"),
        "gemma-3-4b":  ModelConfig("gemma-3-4b",  "fireworks", "accounts/fireworks/models/gemma-3-4b-it", family="gemma"),
        "gemma-3-12b": ModelConfig("gemma-3-12b", "fireworks", "accounts/fireworks/models/gemma-3-12b-it",family="gemma"),
        "gemma-3-27b": ModelConfig("gemma-3-27b", "fireworks", "accounts/fireworks/models/gemma-3-27b-it",family="gemma"),
    },

    "claude": {
        # Claude 3
        "claude-3-haiku":       ModelConfig("claude-3-haiku",       "anthropic", "claude-3-haiku-20240307",      family="claude"),
        "claude-3-sonnet":      ModelConfig("claude-3-sonnet",      "anthropic", "claude-3-sonnet-20240229",     family="claude"),
        "claude-3-opus":        ModelConfig("claude-3-opus",        "anthropic", "claude-3-opus-20240229",       family="claude"),
        # Claude 3.5
        "claude-3.5-haiku":     ModelConfig("claude-3.5-haiku",     "anthropic", "claude-3-5-haiku-20241022",   family="claude"),
        "claude-3.5-sonnet-v1": ModelConfig("claude-3.5-sonnet-v1", "anthropic", "claude-3-5-sonnet-20240620",  family="claude"),
        "claude-3.5-sonnet":    ModelConfig("claude-3.5-sonnet",    "anthropic", "claude-3-5-sonnet-20241022",  family="claude"),
        # Claude 4
        "claude-sonnet-4":      ModelConfig("claude-sonnet-4",      "anthropic", "claude-sonnet-4-20250514",    family="claude"),
        "claude-opus-4":        ModelConfig("claude-opus-4",        "anthropic", "claude-opus-4-20250514",      family="claude"),
        "claude-opus-4.1":      ModelConfig("claude-opus-4.1",      "anthropic", "claude-opus-4-1-20250805",    family="claude"),
        # Claude 4.5
        "claude-haiku-4.5":     ModelConfig("claude-haiku-4.5",     "anthropic", "claude-haiku-4-5-20251001",   family="claude"),
        "claude-sonnet-4.5":    ModelConfig("claude-sonnet-4.5",    "anthropic", "claude-sonnet-4-5-20250929",  family="claude"),
        "claude-opus-4.5":      ModelConfig("claude-opus-4.5",      "anthropic", "claude-opus-4-5-20251101",    family="claude"),
        # Claude 4.6
        "claude-sonnet-4.6":    ModelConfig("claude-sonnet-4.6",    "anthropic", "claude-sonnet-4-6",           family="claude"),
        "claude-opus-4.6":      ModelConfig("claude-opus-4.6",      "anthropic", "claude-opus-4-6",             family="claude"),
    },

    "gpt": {
        # GPT-4o
        "gpt-4o-may24":  ModelConfig("gpt-4o-may24",  "openai", "gpt-4o-2024-05-13",       family="gpt"),
        "gpt-4o-aug24":  ModelConfig("gpt-4o-aug24",  "openai", "gpt-4o-2024-08-06",       family="gpt"),
        "gpt-4o-mini":   ModelConfig("gpt-4o-mini",   "openai", "gpt-4o-mini-2024-07-18",  family="gpt"),
        "gpt-4o-nov24":  ModelConfig("gpt-4o-nov24",  "openai", "gpt-4o-2024-11-20",       family="gpt"),
        # GPT-4.1
        "gpt-4.1":       ModelConfig("gpt-4.1",       "openai", "gpt-4.1-2025-04-14",      family="gpt"),
        "gpt-4.1-mini":  ModelConfig("gpt-4.1-mini",  "openai", "gpt-4.1-mini-2025-04-14", family="gpt"),
        "gpt-4.1-nano":  ModelConfig("gpt-4.1-nano",  "openai", "gpt-4.1-nano-2025-04-14", family="gpt"),
        # o-series reasoning
        "o1":            ModelConfig("o1",      "openai", "o1",             family="gpt", max_tokens=16000),
        "o1-mini":       ModelConfig("o1-mini", "openai", "o1-mini",        family="gpt"),
        "o3-mini":       ModelConfig("o3-mini", "openai", "o3-mini",        family="gpt"),
        "o3":            ModelConfig("o3",      "openai", "o3-2025-04-16",  family="gpt", max_tokens=16000),
        "o4-mini":       ModelConfig("o4-mini", "openai", "o4-mini-2025-04-16", family="gpt"),
        # GPT-5
        "gpt-5":         ModelConfig("gpt-5",      "openai", "gpt-5",      family="gpt"),
        "gpt-5-mini":    ModelConfig("gpt-5-mini", "openai", "gpt-5-mini", family="gpt"),
        "gpt-5.1":       ModelConfig("gpt-5.1",    "openai", "gpt-5.1",    family="gpt"),
        "gpt-5.2":       ModelConfig("gpt-5.2",    "openai", "gpt-5.2",    family="gpt"),
    },

    "gemini": {
        # Gemini 1.5 (legacy, still queryable)
        "gemini-1.5-flash":      ModelConfig("gemini-1.5-flash",      "google", "gemini-1.5-flash-002",  family="gemini"),
        "gemini-1.5-pro":        ModelConfig("gemini-1.5-pro",        "google", "gemini-1.5-pro-002",    family="gemini"),
        # Gemini 2.0
        "gemini-2.0-flash":      ModelConfig("gemini-2.0-flash",      "google", "gemini-2.0-flash",      family="gemini"),
        "gemini-2.0-flash-lite": ModelConfig("gemini-2.0-flash-lite", "google", "gemini-2.0-flash-lite", family="gemini"),
        # Gemini 2.5
        "gemini-2.5-flash-lite": ModelConfig("gemini-2.5-flash-lite", "google", "gemini-2.5-flash-lite", family="gemini"),
        "gemini-2.5-flash":      ModelConfig("gemini-2.5-flash",      "google", "gemini-2.5-flash",      family="gemini"),
        "gemini-2.5-pro":        ModelConfig("gemini-2.5-pro",        "google", "gemini-2.5-pro",         family="gemini"),
        # Gemini 3
        "gemini-3-flash":        ModelConfig("gemini-3-flash",        "google", "gemini-3-flash-preview", family="gemini"),
        "gemini-3-pro":          ModelConfig("gemini-3-pro",          "google", "gemini-3-pro-preview",   family="gemini"),
        "gemini-3.1-pro":        ModelConfig("gemini-3.1-pro",        "google", "gemini-3.1-pro-preview", family="gemini"),
    },
}


DEFAULT_MODELS: Dict[str, ModelConfig] = {
    "llama":  MODEL_REGISTRY["llama"]["llama-3.3-70b"],
    "claude": MODEL_REGISTRY["claude"]["claude-sonnet-4.6"],
    "gpt4":   MODEL_REGISTRY["gpt"]["gpt-4.1"],
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
    """Unified async client for Fireworks AI (open-source), Anthropic (Claude),
    OpenAI (GPT), and Google (Gemini).

    Parameters
    ----------
    models : dict[str, ModelConfig] | None
        Model configs keyed by short name.  Defaults to ``DEFAULT_MODELS``.
    fireworks_api_key, anthropic_api_key, openai_api_key, google_api_key :
        Explicit API keys.  Falls back to the corresponding environment
        variable when *None*.
    """

    def __init__(
        self,
        models: Optional[Dict[str, ModelConfig]] = None,
        fireworks_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.models = models or dict(DEFAULT_MODELS)

        # Fireworks AI (OpenAI-compatible) ---
        from openai import AsyncOpenAI
        fw_key = fireworks_api_key or os.getenv("FIREWORKS_API_KEY")
        if not fw_key:
            raise ValueError(
                "Fireworks API key required via fireworks_api_key param or FIREWORKS_API_KEY env var"
            )
        self._fireworks = AsyncOpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=fw_key,
        )
        self._fireworks_limiter = RateLimiter()

        # Anthropic ---
        from anthropic import AsyncAnthropic
        self._anthropic = AsyncAnthropic(
            api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        # OpenAI ---
        self._openai = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )

        # Google ---
        goog_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if goog_key:
            import google.generativeai as genai
            genai.configure(api_key=goog_key)
        self._google_configured = bool(goog_key)

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _should_retry(error: Exception) -> bool:
        msg = str(error).lower()
        return any(
            kw in msg
            for kw in (
                "rate_limit", "429", "503", "service unavailable",
                "capacity", "timeout", "server error",
            )
        )

    async def _call_fireworks(self, prompt: str, config: ModelConfig) -> str:
        response = await self._fireworks.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        return response.choices[0].message.content

    async def _call_anthropic(self, prompt: str, config: ModelConfig) -> str:
        response = await self._anthropic.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _call_openai(self, prompt: str, config: ModelConfig) -> str:
        response = await self._openai.chat.completions.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def _call_google(self, prompt: str, config: ModelConfig) -> str:
        import google.generativeai as genai
        model = genai.GenerativeModel(config.model_id)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, model.generate_content, prompt)
        return response.text

    # -- multi-message helpers -----------------------------------------------

    async def _call_fireworks_messages(self, messages: List[Dict], config: ModelConfig) -> str:
        response = await self._fireworks.chat.completions.create(
            model=config.model_id,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        return response.choices[0].message.content

    async def _call_anthropic_messages(self, messages: List[Dict], config: ModelConfig) -> str:
        response = await self._anthropic.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=messages,
        )
        return response.content[0].text

    async def _call_openai_messages(self, messages: List[Dict], config: ModelConfig) -> str:
        response = await self._openai.chat.completions.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=messages,
        )
        return response.choices[0].message.content

    async def _call_google_messages(self, messages: List[Dict], config: ModelConfig) -> str:
        import google.generativeai as genai
        model = genai.GenerativeModel(config.model_id)
        # Google uses "model" role instead of "assistant"
        history = []
        for msg in messages[:-1]:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            history.append({"role": role, "parts": [msg["content"]]})
        last_msg = messages[-1]["content"]
        loop = asyncio.get_event_loop()
        chat = model.start_chat(history=history)
        response = await loop.run_in_executor(None, chat.send_message, last_msg)
        return response.text

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

        dispatch = {
            "fireworks": self._call_fireworks_messages,
            "anthropic": self._call_anthropic_messages,
            "openai": self._call_openai_messages,
            "google": self._call_google_messages,
        }

        for name in targets:
            config = self.models[name]
            call_fn = dispatch.get(config.provider)
            if call_fn is None:
                logger.error("Unknown provider %s for model %s", config.provider, name)
                responses[name] = None
                continue

            for attempt in range(max_retries):
                try:
                    responses[name] = await call_fn(messages, config)
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

            if config.provider == "fireworks":
                for attempt in range(max_retries):
                    try:
                        responses[name] = await self._call_fireworks(prompt, config)
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

            elif config.provider == "anthropic":
                for attempt in range(max_retries):
                    try:
                        responses[name] = await self._call_anthropic(prompt, config)
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

            elif config.provider == "openai":
                for attempt in range(max_retries):
                    try:
                        responses[name] = await self._call_openai(prompt, config)
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

            elif config.provider == "google":
                for attempt in range(max_retries):
                    try:
                        responses[name] = await self._call_google(prompt, config)
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
