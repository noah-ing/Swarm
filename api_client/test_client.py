"""Tests for API client module."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from client import APIClient
from config import APIConfig
from exceptions import APIError, APITimeoutError, APIConnectionError


def test_api_config_defaults():
    """Test API configuration defaults."""
    config = APIConfig()
    
    assert config.connection_timeout == 30.0
    assert config.read_timeout == 60.0
    assert config.total_timeout == 120.0
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert config.backoff_factor == 2.0
    assert config.timeout == (30.0, 60.0)
    assert config.default_headers is not None
    assert "application/json" in config.default_headers["Content-Type"]


def test_api_config_with_base_url():
    """Test API configuration with base URL."""
    config = APIConfig()
    new_config = config.with_base_url("https://api.example.com")
    
    assert new_config.base_url == "https://api.example.com"
    assert new_config.connection_timeout == config.connection_timeout


def test_api_config_with_headers():
    """Test API configuration with additional headers."""
    config = APIConfig()
    new_config = config.with_headers({"Authorization": "Bearer token"})
    
    assert "Authorization" in new_config.default_headers
    assert new_config.default_headers["Authorization"] == "Bearer token"
    assert new_config.default_headers["Content-Type"] == "application/json"


def test_api_client_creation():
    """Test API client creation."""
    config = APIConfig()
    client = APIClient(config)
    
    assert client.config == config
    assert client._client is None
    assert client._sync_client is None


def test_build_url():
    """Test URL building."""
    config = APIConfig(base_url="https://api.example.com")
    client = APIClient(config)
    
    url = client._build_url("/users")
    assert url == "https://api.example.com/users"
    
    # Test without base URL
    config_no_base = APIConfig()
    client_no_base = APIClient(config_no_base)
    url = client_no_base._build_url("/users")
    assert url == "/users"


if __name__ == "__main__":
    # Run simple tests
    test_api_config_defaults()
    test_api_config_with_base_url()
    test_api_config_with_headers()
    test_api_client_creation()
    test_build_url()
    print("✓ All tests passed!")