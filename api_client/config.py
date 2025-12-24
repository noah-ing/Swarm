"""API client configuration."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Set, Type

from .retry import RetryConfig, CONSERVATIVE_RETRY


@dataclass
class APIConfig:
    """Configuration for API client timeout and connection settings."""

    # Timeout settings (in seconds)
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    total_timeout: float = 120.0

    # Legacy retry settings (for backwards compatibility)
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0

    # Advanced retry configuration
    retry_config: Optional[RetryConfig] = None

    # Connection settings
    max_connections: int = 10
    max_keepalive_connections: int = 5
    keepalive_expiry: float = 30.0

    # Default headers
    default_headers: Optional[Dict[str, str]] = None

    # Base URL
    base_url: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.default_headers is None:
            self.default_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Swarm-API-Client/1.0"
            }
        
        # If no retry config is provided, create one from legacy settings
        if self.retry_config is None:
            self.retry_config = RetryConfig(
                max_attempts=self.max_retries,
                initial_delay=self.retry_delay,
                backoff_factor=self.backoff_factor,
            )

    @property
    def timeout(self) -> tuple[float, float]:
        """Get timeout tuple for requests (connection, read)."""
        return (self.connection_timeout, self.read_timeout)

    def with_base_url(self, base_url: str) -> "APIConfig":
        """Create a new config with different base URL."""
        return APIConfig(
            connection_timeout=self.connection_timeout,
            read_timeout=self.read_timeout,
            total_timeout=self.total_timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            backoff_factor=self.backoff_factor,
            retry_config=self.retry_config,
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry,
            default_headers=self.default_headers.copy() if self.default_headers else None,
            base_url=base_url
        )

    def with_headers(self, headers: Dict[str, str]) -> "APIConfig":
        """Create a new config with additional headers."""
        new_headers = self.default_headers.copy() if self.default_headers else {}
        new_headers.update(headers)

        return APIConfig(
            connection_timeout=self.connection_timeout,
            read_timeout=self.read_timeout,
            total_timeout=self.total_timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            backoff_factor=self.backoff_factor,
            retry_config=self.retry_config,
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry,
            default_headers=new_headers,
            base_url=self.base_url
        )
    
    def with_retry_config(self, retry_config: RetryConfig) -> "APIConfig":
        """Create a new config with different retry configuration."""
        return APIConfig(
            connection_timeout=self.connection_timeout,
            read_timeout=self.read_timeout,
            total_timeout=self.total_timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            backoff_factor=self.backoff_factor,
            retry_config=retry_config,
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry,
            default_headers=self.default_headers.copy() if self.default_headers else None,
            base_url=self.base_url
        )