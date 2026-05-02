# tests/test_client.py
"""Tests for game_theory_llm.client."""

import time

import pytest

from game_theory_llm.client import DEFAULT_MODELS, MODEL_REGISTRY, ModelConfig, RateLimiter, get_models


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(name="test", provider="openrouter", model_id="openai/gpt-4o")
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.0

    def test_family_field_defaults_empty(self):
        cfg = ModelConfig(name="test", provider="openrouter", model_id="openai/gpt-4o")
        assert cfg.family == ""

    def test_custom_values(self):
        cfg = ModelConfig(
            name="t", provider="openrouter", model_id="meta-llama/llama-3.3-70b-instruct",
            max_tokens=8192, temperature=0.7, family="llama",
        )
        assert cfg.max_tokens == 8192
        assert cfg.temperature == 0.7
        assert cfg.family == "llama"


class TestDefaultModels:
    def test_all_use_openrouter(self):
        providers = {m.provider for m in DEFAULT_MODELS.values()}
        assert providers == {"openrouter"}

    def test_no_legacy_providers(self):
        providers = {m.provider for m in DEFAULT_MODELS.values()}
        for legacy in ("fireworks", "anthropic", "openai", "google", "together"):
            assert legacy not in providers

    def test_claude_model_is_valid(self):
        """Bug #1: Claude model was ' ' (a space). Should be a real model ID."""
        cfg = DEFAULT_MODELS["claude"]
        assert cfg.model_id.strip() != ""
        assert "claude" in cfg.model_id

    def test_has_deepseek_claude_gpt4(self):
        assert "deepseek" in DEFAULT_MODELS
        assert "claude" in DEFAULT_MODELS
        assert "gpt4" in DEFAULT_MODELS

    def test_deepseek_uses_openrouter(self):
        assert DEFAULT_MODELS["deepseek"].provider == "openrouter"


class TestModelRegistry:
    EXPECTED_FAMILIES = {"grok", "qwen", "deepseek", "claude", "gpt", "gemini"}

    def test_has_all_expected_families(self):
        assert self.EXPECTED_FAMILIES <= set(MODEL_REGISTRY.keys())

    def test_each_family_has_at_least_two_models(self):
        for family, models in MODEL_REGISTRY.items():
            assert len(models) >= 2, f"Family '{family}' has fewer than 2 models"

    def test_all_entries_are_model_configs(self):
        for family, models in MODEL_REGISTRY.items():
            for key, cfg in models.items():
                assert isinstance(cfg, ModelConfig), f"{family}/{key} is not a ModelConfig"

    def test_all_registry_entries_have_non_empty_family(self):
        for family, models in MODEL_REGISTRY.items():
            for key, cfg in models.items():
                assert cfg.family != "", f"{family}/{key} has empty family field"

    def test_family_field_matches_registry_key(self):
        for family, models in MODEL_REGISTRY.items():
            for key, cfg in models.items():
                assert cfg.family == family, (
                    f"{family}/{key}: cfg.family={cfg.family!r} != {family!r}"
                )

    def test_all_use_openrouter_provider(self):
        for family, models in MODEL_REGISTRY.items():
            for key, cfg in models.items():
                assert cfg.provider == "openrouter", (
                    f"{family}/{key}: provider={cfg.provider!r} != 'openrouter'"
                )

    def test_all_model_ids_have_slash(self):
        """OpenRouter model IDs use provider/model format."""
        for family, models in MODEL_REGISTRY.items():
            for key, cfg in models.items():
                assert "/" in cfg.model_id, (
                    f"{family}/{key}: model_id={cfg.model_id!r} missing '/' separator"
                )

    def test_get_models_returns_dict_of_configs(self):
        result = get_models("qwen")
        assert isinstance(result, dict)
        assert len(result) > 0
        for cfg in result.values():
            assert isinstance(cfg, ModelConfig)

    def test_get_models_unknown_family_returns_empty(self):
        assert get_models("nonexistent") == {}

    def test_default_models_reference_registry_entries(self):
        assert DEFAULT_MODELS["deepseek"] is MODEL_REGISTRY["deepseek"]["deepseek-v3"]
        assert DEFAULT_MODELS["claude"] is MODEL_REGISTRY["claude"]["claude-sonnet-4.6"]
        assert DEFAULT_MODELS["gpt4"] is MODEL_REGISTRY["gpt"]["gpt-4.1"]


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_first_request(self):
        rl = RateLimiter()
        ok, wait = await rl.enforce(100)
        assert ok is True
        assert wait == 0

    @pytest.mark.asyncio
    async def test_tracks_timestamps(self):
        rl = RateLimiter()
        await rl.enforce(0)
        assert len(rl.request_timestamps) == 1

    @pytest.mark.asyncio
    async def test_tracks_token_usage(self):
        rl = RateLimiter()
        await rl.enforce(500)
        assert len(rl.token_usage) == 1
        assert rl.token_usage[0][1] == 500

    def test_update_token_usage(self):
        rl = RateLimiter()
        rl.token_usage = [(time.time(), 100)]
        rl.update_token_usage(250)
        assert rl.token_usage[0][1] == 250
