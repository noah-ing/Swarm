"""Configuration management for Swarm."""

import os
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Swarm configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # Provider preference
    prefer_provider: Literal["anthropic", "openai"] = "anthropic"

    # Model defaults
    default_model: str = "sonnet"

    # Model mappings
    @property
    def model_map(self) -> dict[str, str]:
        """Map friendly names to actual model IDs."""
        return {
            # Anthropic
            "haiku": "claude-3-5-haiku-latest",
            "sonnet": "claude-sonnet-4-20250514",
            "opus": "claude-opus-4-20250514",
            # OpenAI
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4o": "gpt-4o",
            "o1": "o1",
            "o1-mini": "o1-mini",
        }

    def get_model_id(self, friendly_name: str) -> str:
        """Convert friendly name to model ID."""
        return self.model_map.get(friendly_name, friendly_name)

    def get_provider(self, model: str) -> Literal["anthropic", "openai"]:
        """Determine provider from model name."""
        if model in ("haiku", "sonnet", "opus") or model.startswith("claude"):
            return "anthropic"
        return "openai"

    # Tool settings
    bash_timeout: int = 120  # seconds
    max_output_length: int = 50000  # characters

    # Agent settings
    max_retries: int = 3
    max_iterations: int = 50  # per grunt

    # Cost tracking (per 1M tokens)
    cost_per_million: dict[str, dict[str, float]] = {
        "haiku": {"input": 0.25, "output": 1.25},
        "sonnet": {"input": 3.0, "output": 15.0},
        "opus": {"input": 15.0, "output": 75.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.0},
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
