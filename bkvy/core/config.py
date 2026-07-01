"""
Configuration management for bkvy
"""

import json
import aiofiles
from pathlib import Path
from typing import Dict, List, Tuple, Any

from ..models.data_classes import ProviderConfig, ProviderKey, ProviderModel
from ..utils.logging import setup_logging

logger = setup_logging()

VALID_TIERS = {"low", "medium", "high"}


class ConfigManager:
    """Manages loading, validating, and hot-reloading of configuration files.

    Startup load raises on invalid config (fail fast, with precise errors).
    Hot reload (refresh_if_changed) never raises: a bad edit is logged and the
    last known-good configuration stays active, so one typo cannot take down
    every endpoint.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.providers: Dict[str, ProviderConfig] = {}
        self.scenarios: Dict[str, List[Dict[str, Any]]] = {}
        self._last_providers_mtime = 0
        self._last_routing_mtime = 0

    async def load_configs(self):
        """Load all configuration files (raises on invalid config)"""
        await self._load_providers()
        await self._load_scenarios()

    @staticmethod
    def _validate_providers(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate the raw providers.json structure.

        Returns (errors, warnings). Errors make routing incorrect and reject
        the config; warnings flag suspicious but workable entries.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(data, dict) or not data:
            return (["providers.json must be a non-empty JSON object"], [])

        for provider_name, provider_data in data.items():
            if not isinstance(provider_data, dict):
                errors.append(f"{provider_name}: must be an object")
                continue

            models = provider_data.get("models", {})
            keys = provider_data.get("keys", {})

            if not isinstance(models, dict) or not models:
                errors.append(f"{provider_name}: 'models' must be a non-empty object")
                models = {}
            if not isinstance(keys, dict) or not keys:
                errors.append(f"{provider_name}: 'keys' must be a non-empty object")
                keys = {}

            for model_name, model_data in models.items():
                loc = f"{provider_name}.models.{model_name}"
                if not isinstance(model_data, dict):
                    errors.append(f"{loc}: must be an object")
                    continue
                endpoint = model_data.get("endpoint")
                if not isinstance(endpoint, str) or not endpoint.strip():
                    errors.append(f"{loc}: 'endpoint' must be a non-empty string")
                tier = model_data.get("intelligence_tier")
                if tier not in VALID_TIERS:
                    errors.append(f"{loc}: 'intelligence_tier' must be one of {sorted(VALID_TIERS)}, got {tier!r}")
                cost = model_data.get("cost_per_1k_tokens")
                if not isinstance(cost, (int, float)) or cost < 0:
                    errors.append(f"{loc}: 'cost_per_1k_tokens' must be a non-negative number")
                avg_ms = model_data.get("avg_response_time_ms")
                if not isinstance(avg_ms, int) or avg_ms <= 0:
                    errors.append(f"{loc}: 'avg_response_time_ms' must be a positive integer")

            covered_models = set()
            for key_id, key_data in keys.items():
                loc = f"{provider_name}.keys.{key_id}"
                if not isinstance(key_data, dict):
                    errors.append(f"{loc}: must be an object")
                    continue
                api_key = key_data.get("api_key")
                if not isinstance(api_key, str):
                    errors.append(f"{loc}: 'api_key' must be a string")
                elif not api_key.strip() and provider_name != "ollama":
                    warnings.append(f"{loc}: 'api_key' is empty")
                rate_limits = key_data.get("rate_limits")
                if not isinstance(rate_limits, dict) or not rate_limits:
                    errors.append(f"{loc}: 'rate_limits' must be a non-empty object")
                    continue
                for rl_model, rl in rate_limits.items():
                    rl_loc = f"{loc}.rate_limits.{rl_model}"
                    if rl_model not in models:
                        warnings.append(f"{rl_loc}: model not defined in {provider_name}.models (entry is ignored)")
                    else:
                        covered_models.add(rl_model)
                    if not isinstance(rl, dict) or not isinstance(rl.get("rpm"), int) \
                            or not isinstance(rl.get("rpd"), int) or rl.get("rpm", -1) < 0 \
                            or rl.get("rpd", -1) < 0:
                        errors.append(f"{rl_loc}: must contain non-negative integer 'rpm' and 'rpd'")

            for model_name in models:
                if model_name not in covered_models:
                    warnings.append(
                        f"{provider_name}.models.{model_name}: no key has rate_limits for this model "
                        f"- it can never be routed to")

        return errors, warnings

    def _validate_scenarios(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate routing.json scenarios against the loaded provider config"""
        errors: List[str] = []
        warnings: List[str] = []

        scenarios = data.get("scenarios", {})
        if not isinstance(scenarios, dict):
            return (["routing.json: 'scenarios' must be an object"], [])

        for name, items in scenarios.items():
            if not isinstance(items, list) or not items:
                errors.append(f"scenario '{name}': must be a non-empty list")
                continue
            for i, item in enumerate(items):
                loc = f"scenario '{name}'[{i}]"
                if not isinstance(item, dict):
                    errors.append(f"{loc}: must be an object")
                    continue
                provider = item.get("provider")
                model = item.get("model")
                if not isinstance(item.get("priority"), int):
                    errors.append(f"{loc}: 'priority' must be an integer")
                # Stale references are warnings, not errors: the entry is
                # skipped at runtime (as before), but now it is surfaced
                # loudly instead of failing silently forever.
                if provider not in self.providers:
                    warnings.append(f"{loc}: unknown provider {provider!r} (entry is ignored)")
                elif model not in self.providers[provider].models:
                    warnings.append(f"{loc}: unknown model {model!r} for provider {provider!r} (entry is ignored)")

        return errors, warnings

    async def _load_providers(self):
        """Load and validate providers.json configuration (raises on failure)"""
        providers_file = self.config_dir / "providers.json"

        if not providers_file.exists():
            logger.error("providers.json not found", file_path=str(providers_file))
            raise FileNotFoundError(f"Configuration file not found: {providers_file}")

        current_mtime = providers_file.stat().st_mtime
        if current_mtime == self._last_providers_mtime:
            return  # No changes

        # Record the attempt up front: a broken file is not retried (and
        # re-logged) on every request, only when it changes again.
        self._last_providers_mtime = current_mtime

        async with aiofiles.open(providers_file, 'r') as f:
            content = await f.read()
            data = json.loads(content)

        errors, warnings = self._validate_providers(data)
        for warning in warnings:
            logger.warning("providers.json validation warning", detail=warning)
        if errors:
            raise ValueError("providers.json validation failed: " + " | ".join(errors))

        # Parse into a fresh dict and swap atomically, so a reload can never
        # leave a half-applied config and removed providers actually disappear.
        new_providers: Dict[str, ProviderConfig] = {}
        for provider_name, provider_data in data.items():
            keys = {}
            for key_id, key_data in provider_data.get("keys", {}).items():
                keys[key_id] = ProviderKey(
                    api_key=key_data["api_key"],
                    rate_limits=key_data["rate_limits"]
                )

            models = {}
            for model_name, model_data in provider_data.get("models", {}).items():
                models[model_name] = ProviderModel(
                    endpoint=model_data["endpoint"],
                    cost_per_1k_tokens=model_data["cost_per_1k_tokens"],
                    avg_response_time_ms=model_data["avg_response_time_ms"],
                    intelligence_tier=model_data["intelligence_tier"],
                    version=model_data.get("version"),
                    supports_thinking=model_data.get("supports_thinking")
                )

            new_providers[provider_name] = ProviderConfig(keys=keys, models=models)

        self.providers = new_providers
        logger.info("Loaded providers configuration", provider_count=len(self.providers))

    async def _load_scenarios(self):
        """Load and validate routing.json configuration (raises on failure)"""
        routing_file = self.config_dir / "routing.json"

        if not routing_file.exists():
            logger.error("routing.json not found", file_path=str(routing_file))
            raise FileNotFoundError(f"Configuration file not found: {routing_file}")

        current_mtime = routing_file.stat().st_mtime
        if current_mtime == self._last_routing_mtime:
            return  # No changes

        self._last_routing_mtime = current_mtime

        async with aiofiles.open(routing_file, 'r') as f:
            content = await f.read()
            data = json.loads(content)

        errors, warnings = self._validate_scenarios(data)
        for warning in warnings:
            logger.warning("routing.json validation warning", detail=warning)
        if errors:
            raise ValueError("routing.json validation failed: " + " | ".join(errors))

        self.scenarios = data.get("scenarios", {})
        logger.info("Loaded routing scenarios", scenario_count=len(self.scenarios))

    async def refresh_if_changed(self):
        """Check for configuration file changes and reload if necessary.

        Never raises: on a bad edit the previous configuration stays active.
        """
        try:
            await self._load_providers()
        except Exception as e:
            logger.error("Config hot-reload failed for providers.json - keeping previous configuration",
                        error=str(e))
        try:
            await self._load_scenarios()
        except Exception as e:
            logger.error("Config hot-reload failed for routing.json - keeping previous configuration",
                        error=str(e))

    def get_models_by_intelligence(self, intelligence_level: str) -> List[Tuple[str, str]]:
        """Get all (provider, model) combinations for an intelligence level"""
        combinations = []

        for provider_name, config in self.providers.items():
            for model_name, model in config.models.items():
                if model.intelligence_tier == intelligence_level:
                    combinations.append((provider_name, model_name))

        logger.debug("Intelligence level search complete",
                    intelligence_level=intelligence_level,
                    total_combinations=len(combinations),
                    combinations=combinations)

        return combinations

    def get_scenario_combinations(self, scenario_name: str) -> List[Tuple[str, str, int]]:
        """Get (provider, model, priority) combinations for a scenario"""
        if scenario_name not in self.scenarios:
            return []

        combinations = []
        for item in self.scenarios[scenario_name]:
            combinations.append((item["provider"], item["model"], item["priority"]))
        return combinations
