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


class ConfigManager:
    """Manages loading and hot-reloading of configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.providers: Dict[str, ProviderConfig] = {}
        self.scenarios: Dict[str, List[Dict[str, Any]]] = {}
        self._last_providers_mtime = 0
        self._last_routing_mtime = 0
    
    async def load_configs(self):
        """Load all configuration files"""
        await self._load_providers()
        await self._load_scenarios()
    
    async def _load_providers(self):
        """Load providers.json configuration"""
        providers_file = self.config_dir / "providers.json"
        
        if not providers_file.exists():
            logger.error("providers.json not found", file_path=str(providers_file))
            raise FileNotFoundError(f"Configuration file not found: {providers_file}")
        
        current_mtime = providers_file.stat().st_mtime
        if current_mtime == self._last_providers_mtime:
            return  # No changes
        
        try:
            async with aiofiles.open(providers_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            # Parse provider configurations
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
                
                self.providers[provider_name] = ProviderConfig(keys=keys, models=models)
            
            self._last_providers_mtime = current_mtime
            logger.info("Loaded providers configuration", provider_count=len(self.providers))
            
        except Exception as e:
            logger.error("Failed to load providers configuration", error=str(e))
            raise
    
    async def _load_scenarios(self):
        """Load routing.json configuration"""
        routing_file = self.config_dir / "routing.json"
        
        if not routing_file.exists():
            logger.error("routing.json not found", file_path=str(routing_file))
            raise FileNotFoundError(f"Configuration file not found: {routing_file}")
        
        current_mtime = routing_file.stat().st_mtime
        if current_mtime == self._last_routing_mtime:
            return  # No changes
        
        try:
            async with aiofiles.open(routing_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            self.scenarios = data.get("scenarios", {})
            self._last_routing_mtime = current_mtime
            logger.info("Loaded routing scenarios", scenario_count=len(self.scenarios))
            
        except Exception as e:
            logger.error("Failed to load routing configuration", error=str(e))
            raise
    
    async def refresh_if_changed(self):
        """Check for configuration file changes and reload if necessary"""
        await self._load_providers()
        await self._load_scenarios()
    
    def get_models_by_intelligence(self, intelligence_level: str) -> List[Tuple[str, str]]:
        """Get all (provider, model) combinations for an intelligence level"""
        combinations = []
        
        logger.info("🔍 CONFIG DEBUG: Looking for models with intelligence level", 
                   intelligence_level=intelligence_level)
        
        for provider_name, config in self.providers.items():
            logger.info("🔍 CONFIG DEBUG: Checking provider", 
                       provider=provider_name,
                       model_count=len(config.models))
            
            for model_name, model in config.models.items():
                logger.info("🔍 CONFIG DEBUG: Checking model", 
                           provider=provider_name,
                           model=model_name,
                           model_intelligence_tier=model.intelligence_tier,
                           matches=model.intelligence_tier == intelligence_level)
                
                if model.intelligence_tier == intelligence_level:
                    combinations.append((provider_name, model_name))
                    logger.info("✅ CONFIG DEBUG: Model added to combinations", 
                               provider=provider_name,
                               model=model_name,
                               intelligence_tier=model.intelligence_tier)
        
        logger.info("🏁 CONFIG DEBUG: Intelligence level search complete", 
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