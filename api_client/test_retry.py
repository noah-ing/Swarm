"""Test retry logic with various configurations."""

import asyncio
import time
from typing import Optional

import httpx

from api_client import (
    APIClient,
    APIConfig,
    RetryConfig,
    AGGRESSIVE_RETRY,
    RATE_LIMIT_RETRY,
    APITimeoutError,
    APIRateLimitError,
    ServiceUnavailableError,
)


def test_basic_retry():
    """Test basic retry with exponential backoff."""
    print("\n=== Testing Basic Retry with Exponential Backoff ===")
    
    # Custom retry config with monitoring
    def on_retry(attempt: int, exception: Exception, delay: float):
        print(f"Attempt {attempt} failed: {type(exception).__name__}: {exception}")
        print(f"Retrying in {delay:.2f} seconds...")
    
    def on_give_up(exception: Exception):
        print(f"Giving up after retries: {type(exception).__name__}: {exception}")
    
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        jitter=True,
        on_retry=on_retry,
        on_give_up=on_give_up,
    )
    
    config = APIConfig(retry_config=retry_config)
    client = APIClient(config)
    
    try:
        # This will fail but demonstrate retry behavior
        response = client.get("https://httpbin.org/status/503")
        print(f"Unexpected success: {response}")
    except ServiceUnavailableError as e:
        print(f"\nFinal error after retries: {type(e).__name__}")
        print(f"Error details: {e}")


def test_rate_limit_retry():
    """Test retry with rate limit handling."""
    print("\n\n=== Testing Rate Limit Retry ===")
    
    def on_retry(attempt: int, exception: Exception, delay: float):
        print(f"Rate limit hit (attempt {attempt})")
        if hasattr(exception, 'context') and isinstance(exception.context, dict):
            if 'retry_after' in exception.context:
                print(f"Server says retry after: {exception.context['retry_after']}s")
        print(f"Waiting {delay:.2f} seconds...")
    
    # Use preset rate limit retry config with monitoring
    rate_limit_config = RetryConfig(
        **RATE_LIMIT_RETRY.__dict__,
        on_retry=on_retry,
    )
    
    config = APIConfig(retry_config=rate_limit_config)
    client = APIClient(config)
    
    try:
        # Simulate rate limit
        response = client.get("https://httpbin.org/status/429")
        print(f"Unexpected success: {response}")
    except APIRateLimitError as e:
        print(f"\nFinal error: {type(e).__name__}")
        print(f"Rate limit details: {e.context}")


def test_timeout_retry():
    """Test retry on timeout errors."""
    print("\n\n=== Testing Timeout Retry ===")
    
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay=0.5,
        backoff_factor=2.0,
        jitter=True,
        retry_on_exceptions={APITimeoutError, httpx.TimeoutException},
    )
    
    # Very short timeout to trigger timeouts
    config = APIConfig(
        connection_timeout=0.001,  # 1ms - will timeout
        retry_config=retry_config,
    )
    client = APIClient(config)
    
    try:
        # This should timeout and retry
        response = client.get("https://httpbin.org/delay/1")
        print(f"Unexpected success: {response}")
    except APITimeoutError as e:
        print(f"Timeout error after retries: {e}")


def test_custom_retry_condition():
    """Test custom retry conditions."""
    print("\n\n=== Testing Custom Retry Condition ===")
    
    # Only retry on specific error messages
    def custom_condition(exception: Exception) -> bool:
        if hasattr(exception, 'context') and isinstance(exception.context, dict):
            error_msg = str(exception.context.get('response', {}).get('error', ''))
            return 'temporary' in error_msg.lower() or 'retry' in error_msg.lower()
        return False
    
    retry_config = RetryConfig(
        max_attempts=3,
        custom_retry_condition=custom_condition,
        retry_on_exceptions=set(),  # Clear default exceptions
        retry_on_status_codes=set(),  # Clear default status codes
    )
    
    config = APIConfig(retry_config=retry_config)
    client = APIClient(config)
    
    print("Testing error that should NOT retry...")
    try:
        response = client.get("https://httpbin.org/status/404")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Got error (no retry): {type(e).__name__}")


def test_max_retry_time():
    """Test maximum retry time limit."""
    print("\n\n=== Testing Maximum Retry Time ===")
    
    retry_attempts = 0
    
    def on_retry(attempt: int, exception: Exception, delay: float):
        nonlocal retry_attempts
        retry_attempts = attempt
        print(f"Attempt {attempt} at {time.time():.2f}")
    
    retry_config = RetryConfig(
        max_attempts=10,  # High number of attempts
        initial_delay=1.0,
        backoff_factor=2.0,
        max_retry_time=5.0,  # But only 5 seconds total
        on_retry=on_retry,
    )
    
    config = APIConfig(retry_config=retry_config)
    client = APIClient(config)
    
    start_time = time.time()
    try:
        response = client.get("https://httpbin.org/status/503")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nStopped after {retry_attempts} attempts in {elapsed:.2f} seconds")
        print(f"(configured max time: {retry_config.max_retry_time}s)")


async def test_async_retry():
    """Test async retry functionality."""
    print("\n\n=== Testing Async Retry ===")
    
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay=0.5,
        jitter=True,
    )
    
    config = APIConfig(retry_config=retry_config)
    client = APIClient(config)
    
    try:
        async with client:
            response = await client.aget("https://httpbin.org/status/500")
            print(f"Response: {response}")
    except Exception as e:
        print(f"Async error after retries: {type(e).__name__}")


def test_no_jitter():
    """Test retry without jitter."""
    print("\n\n=== Testing Retry Without Jitter ===")
    
    delays = []
    
    def on_retry(attempt: int, exception: Exception, delay: float):
        delays.append(delay)
        print(f"Attempt {attempt}: delay = {delay:.3f}s")
    
    retry_config = RetryConfig(
        max_attempts=4,
        initial_delay=1.0,
        backoff_factor=2.0,
        jitter=False,  # No randomness
        on_retry=on_retry,
    )
    
    config = APIConfig(retry_config=retry_config)
    client = APIClient(config)
    
    try:
        response = client.get("https://httpbin.org/status/503")
    except:
        print("\nDelays should follow pattern: 1, 2, 4 (initial * factor^n)")
        print(f"Actual delays: {[f'{d:.3f}' for d in delays]}")


def main():
    """Run all retry tests."""
    print("Testing Retry Logic with Exponential Backoff")
    print("=" * 50)
    
    test_basic_retry()
    test_rate_limit_retry()
    test_timeout_retry()
    test_custom_retry_condition()
    test_max_retry_time()
    test_no_jitter()
    
    # Run async test
    print("\nRunning async test...")
    asyncio.run(test_async_retry())
    
    print("\n\nAll retry tests completed!")


if __name__ == "__main__":
    main()