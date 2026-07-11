"""Model registry - loads config and creates adapters."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from secdrift.models.base import ModelAdapter
from secdrift.models.providers import (
    BedrockAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    OpenAICompatibleAdapter,
    PROVIDER_CLASSES,
)


# Default config location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"


def _expand_env_vars(value: Any) -> Any:
    """Expand environment variables in config values.

    Supports: ${VAR}, ${VAR:-default}, ${VAR-default}
    """
    if isinstance(value, str):
        # Pattern: ${VAR:-default} or ${VAR-default} or ${VAR}
        pattern = r'\$\{([^}:]+)(?::-?([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(var_name, default)

        return re.sub(pattern, replacer, value)

    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]

    return value


class ModelConfig:
    """Loaded model configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load and parse config file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f)

        # Expand environment variables
        self._config = _expand_env_vars(raw_config)

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._config.get("defaults", {})

    @property
    def providers(self) -> Dict[str, Dict[str, Any]]:
        return self._config.get("providers", {})

    @property
    def models(self) -> Dict[str, Dict[str, Any]]:
        return self._config.get("models", {})

    @property
    def groups(self) -> Dict[str, List[str]]:
        return self._config.get("groups", {})

    @property
    def active_group(self) -> Optional[str]:
        return self._config.get("active", {}).get("group")

    @property
    def active_models(self) -> Optional[List[str]]:
        return self._config.get("active", {}).get("models")

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)

    def get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific provider."""
        return self.providers.get(provider_name)

    def get_group_models(self, group_name: str) -> List[str]:
        """Get model names in a group."""
        return self.groups.get(group_name, [])

    def get_enabled_models(self) -> List[str]:
        """Get all enabled model names."""
        return [
            name for name, config in self.models.items()
            if config.get("enabled", False)
        ]


# Global config instance
_config: Optional[ModelConfig] = None


def get_config(config_path: Optional[Path] = None) -> ModelConfig:
    """Get or create the model configuration."""
    global _config
    if _config is None or config_path is not None:
        _config = ModelConfig(config_path)
    return _config


def get_model(
    model_name: str,
    config_path: Optional[Path] = None,
    **override_kwargs
) -> ModelAdapter:
    """Create a model adapter by name.

    Args:
        model_name: Model name from config (e.g., 'claude-3-5-sonnet')
                   Or explicit provider format: 'bedrock/model-id'
        config_path: Optional path to config file
        **override_kwargs: Override any config values

    Returns:
        Configured ModelAdapter instance

    Examples:
        >>> model = get_model("claude-3-5-sonnet")
        >>> model = get_model("llama-3-1-70b", temperature=0.5)
        >>> model = get_model("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    """
    config = get_config(config_path)

    # Check for explicit provider prefix (e.g., "bedrock/model-id")
    if "/" in model_name and model_name.split("/")[0] in config.providers:
        parts = model_name.split("/", 1)
        provider_name = parts[0]
        model_id = parts[1]
        model_config = {
            "provider": provider_name,
            "model_id": model_id,
        }
    else:
        # Look up in config
        model_config = config.get_model_config(model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(config.models.keys())}")

    # Get provider config
    provider_name = model_config["provider"]
    provider_config = config.get_provider_config(provider_name) or {}

    # Determine adapter class
    provider_type = provider_config.get("type", provider_name)
    adapter_class = PROVIDER_CLASSES.get(provider_type)

    if adapter_class is None:
        raise ValueError(f"Unknown provider type: {provider_type}")

    # Build adapter kwargs
    kwargs = {
        "model_id": model_config.get("model_id", model_name),
        "temperature": config.defaults.get("temperature", 0.7),
        "max_tokens": config.defaults.get("max_tokens", 2048),
        "timeout": config.defaults.get("timeout_seconds", 90),
        "max_retries": config.defaults.get("max_retries", 3),
    }

    # Add provider-specific config
    if provider_type == "bedrock":
        kwargs["region"] = provider_config.get("region", "us-east-1")
        profile = provider_config.get("profile", "")
        if profile:
            kwargs["profile"] = profile

    elif provider_type in ("openai", "openai_compatible"):
        kwargs["api_key"] = provider_config.get("api_key")
        kwargs["base_url"] = provider_config.get("base_url", "https://api.openai.com/v1")
        if provider_type == "openai_compatible":
            kwargs["provider_name"] = provider_name  # For logging

    elif provider_type == "anthropic":
        kwargs["api_key"] = provider_config.get("api_key")

    # Apply model-specific params
    if "params" in model_config:
        kwargs.update(model_config["params"])

    # Apply overrides
    kwargs.update(override_kwargs)

    return adapter_class(**kwargs)


def get_models(
    model_names: Optional[List[str]] = None,
    group: Optional[str] = None,
    config_path: Optional[Path] = None,
    **override_kwargs
) -> List[ModelAdapter]:
    """Create multiple model adapters.

    Args:
        model_names: List of model names to create
        group: Group name to use (ignored if model_names provided)
        config_path: Optional path to config file
        **override_kwargs: Override any config values for all models

    Returns:
        List of configured ModelAdapter instances
    """
    config = get_config(config_path)

    if model_names is None:
        if group:
            model_names = config.get_group_models(group)
        elif config.active_models:
            model_names = config.active_models
        elif config.active_group:
            model_names = config.get_group_models(config.active_group)
        else:
            model_names = config.get_enabled_models()

    return [get_model(name, config_path, **override_kwargs) for name in model_names]


def list_models(config_path: Optional[Path] = None) -> List[str]:
    """List all available model names."""
    config = get_config(config_path)
    return list(config.models.keys())


def list_providers(config_path: Optional[Path] = None) -> List[str]:
    """List all configured providers."""
    config = get_config(config_path)
    return list(config.providers.keys())


def list_groups(config_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """List all model groups."""
    config = get_config(config_path)
    return config.groups
