"""Config validation and safe hot reload."""

import asyncio
import json

import pytest

from bkvy.core.config import ConfigManager


def run(coro):
    return asyncio.run(coro)


GOOD_PROVIDERS = {
    "openai": {
        "keys": {
            "k1": {"api_key": "sk-x", "rate_limits": {"gpt": {"rpm": 10, "rpd": 100}}}
        },
        "models": {
            "gpt": {"endpoint": "https://api.openai.com/v1/chat/completions",
                    "cost_per_1k_tokens": 0.001, "avg_response_time_ms": 1500,
                    "intelligence_tier": "low"}
        },
    }
}

GOOD_ROUTING = {"scenarios": {"default": [{"provider": "openai", "model": "gpt", "priority": 1}]}}


def write_config(tmp_path, providers=GOOD_PROVIDERS, routing=GOOD_ROUTING):
    (tmp_path / "providers.json").write_text(json.dumps(providers))
    (tmp_path / "routing.json").write_text(json.dumps(routing))


class TestValidation:
    def test_good_config_loads(self, tmp_path):
        write_config(tmp_path)
        cm = ConfigManager(config_dir=str(tmp_path))
        run(cm.load_configs())
        assert "openai" in cm.providers
        assert "default" in cm.scenarios

    def test_bad_tier_rejected(self, tmp_path):
        bad = json.loads(json.dumps(GOOD_PROVIDERS))
        bad["openai"]["models"]["gpt"]["intelligence_tier"] = "ultra"
        write_config(tmp_path, providers=bad)
        cm = ConfigManager(config_dir=str(tmp_path))
        with pytest.raises(ValueError, match="intelligence_tier"):
            run(cm.load_configs())

    def test_missing_endpoint_rejected(self, tmp_path):
        bad = json.loads(json.dumps(GOOD_PROVIDERS))
        del bad["openai"]["models"]["gpt"]["endpoint"]
        write_config(tmp_path, providers=bad)
        cm = ConfigManager(config_dir=str(tmp_path))
        with pytest.raises(ValueError, match="endpoint"):
            run(cm.load_configs())

    def test_bad_rate_limits_rejected(self, tmp_path):
        bad = json.loads(json.dumps(GOOD_PROVIDERS))
        bad["openai"]["keys"]["k1"]["rate_limits"]["gpt"] = {"rpm": "ten", "rpd": 100}
        write_config(tmp_path, providers=bad)
        cm = ConfigManager(config_dir=str(tmp_path))
        with pytest.raises(ValueError, match="rpm"):
            run(cm.load_configs())

    def test_stale_scenario_reference_is_warning_not_error(self, tmp_path):
        routing = {"scenarios": {"default": [
            {"provider": "openai", "model": "gpt", "priority": 1},
            {"provider": "openai", "model": "gone-model", "priority": 2},
        ]}}
        write_config(tmp_path, routing=routing)
        cm = ConfigManager(config_dir=str(tmp_path))
        run(cm.load_configs())  # must not raise
        assert "default" in cm.scenarios


class TestHotReload:
    def test_bad_edit_keeps_last_good_config(self, tmp_path):
        write_config(tmp_path)
        cm = ConfigManager(config_dir=str(tmp_path))
        run(cm.load_configs())
        assert "openai" in cm.providers

        # Break the file (invalid JSON), bump mtime
        providers_file = tmp_path / "providers.json"
        providers_file.write_text("{ not json")
        import os
        os.utime(providers_file, (providers_file.stat().st_atime,
                                  providers_file.stat().st_mtime + 10))

        run(cm.refresh_if_changed())  # must not raise
        assert "openai" in cm.providers  # previous config still active

    def test_valid_edit_applies_and_removes_deleted_providers(self, tmp_path):
        write_config(tmp_path)
        cm = ConfigManager(config_dir=str(tmp_path))
        run(cm.load_configs())

        new_providers = json.loads(json.dumps(GOOD_PROVIDERS))
        new_providers["anthropic"] = new_providers.pop("openai")
        providers_file = tmp_path / "providers.json"
        providers_file.write_text(json.dumps(new_providers))
        import os
        os.utime(providers_file, (providers_file.stat().st_atime,
                                  providers_file.stat().st_mtime + 10))

        run(cm.refresh_if_changed())
        assert "anthropic" in cm.providers
        assert "openai" not in cm.providers  # swap, not merge
