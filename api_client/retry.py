"""Advanced retry logic with exponential backoff and configurable conditions."""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Type, Union, Any, Dict
from functools import wraps

import httpx

from .exceptions import (
    APIError,
    APITimeoutError,
    APIConnectionError,
    APIRateLimitError,
    NetworkError,
    HTTPServerError,
    ServiceUnavailableError,
    BadGatewayError,
    GatewayTimeoutError,
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""
    
    # Basic retry settings
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    
    # Jitter settings (to avoid thundering herd)
    jitter: bool = True
    jitter_factor: float = 0.1  # +/- 10% randomness
    
    # Maximum total time to spend retrying
    max_retry_time: Optional[float] = None
    
    # Retry conditions
    retry_on_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {
        APITimeoutError,
        APIConnectionError,
        NetworkError,
        ServiceUnavailableError,
        BadGatewayError,
        GatewayTimeoutError,
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.NetworkError,
    })
    
    retry_on_status_codes: Set[int] = field(default_factory=lambda: {
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    })
    
    # Special handling for rate limits
    respect_retry_after_header: bool = True
    
    # Custom retry condition function
    custom_retry_condition: Optional[Callable[[Exception], bool]] = None
    
    # Callbacks for monitoring
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_give_up: Optional[Callable[[Exception], None]] = None
    
    def should_retry(self, exception: Exception) -> bool:
        """Check if the exception warrants a retry."""
        # Check custom condition first
        if self.custom_retry_condition and self.custom_retry_condition(exception):
            return True
        
        # Check exception type
        for exc_type in self.retry_on_exceptions:
            if isinstance(exception, exc_type):
                return True
                
        # Check status code if it's an API error
        if hasattr(exception, 'status_code'):
            if exception.status_code in self.retry_on_status_codes:
                return True
                
        return False
    
    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Calculate the delay before the next retry attempt."""
        # Use Retry-After header if available and configured
        if retry_after is not None and self.respect_retry_after_header:
            base_delay = retry_after
        else:
            # Exponential backoff: initial_delay * (backoff_factor ^ (attempt - 1))
            base_delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        
        # Apply maximum delay cap
        base_delay = min(base_delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter and base_delay > 0:
            jitter_range = base_delay * self.jitter_factor
            jitter = random.uniform(-jitter_range, jitter_range)
            base_delay = max(0, base_delay + jitter)
        
        return base_delay


class RetryManager:
    """Manager for handling retries with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry manager with configuration."""
        self.config = config or RetryConfig()
        self._start_time: Optional[float] = None
    
    def _extract_retry_after(self, exception: Exception) -> Optional[float]:
        """Extract Retry-After value from exception if available."""
        if isinstance(exception, APIRateLimitError):
            # Check for retry_after in exception context
            if hasattr(exception, 'context') and isinstance(exception.context, dict):
                return exception.context.get('retry_after')
        
        # Check for Retry-After in response headers
        if hasattr(exception, 'response') and hasattr(exception.response, 'headers'):
            retry_after_header = exception.response.headers.get('Retry-After')
            if retry_after_header:
                try:
                    return float(retry_after_header)
                except ValueError:
                    # Might be a date string, ignore for now
                    pass
        
        return None
    
    def _check_time_limit(self) -> bool:
        """Check if we've exceeded the maximum retry time."""
        if self.config.max_retry_time is None:
            return True
            
        if self._start_time is None:
            return True
            
        elapsed = time.time() - self._start_time
        return elapsed < self.config.max_retry_time
    
    def execute_with_retry(self, func: Callable[[], Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a function with retry logic."""
        self._start_time = time.time()
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if not self.config.should_retry(e):
                    if self.config.on_give_up:
                        self.config.on_give_up(e)
                    raise
                
                # Check if this is the last attempt
                if attempt >= self.config.max_attempts:
                    if self.config.on_give_up:
                        self.config.on_give_up(e)
                    raise
                
                # Check time limit
                if not self._check_time_limit():
                    if self.config.on_give_up:
                        self.config.on_give_up(e)
                    raise
                
                # Calculate retry delay
                retry_after = self._extract_retry_after(e)
                delay = self.config.calculate_delay(attempt, retry_after)
                
                # Call retry callback if configured
                if self.config.on_retry:
                    self.config.on_retry(attempt, e, delay)
                
                # Wait before retry
                time.sleep(delay)
        
        # This shouldn't happen, but just in case
        if last_exception:
            raise last_exception
        raise APIError("Retry logic error: no exception captured")
    
    async def async_execute_with_retry(self, func: Callable[[], Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute an async function with retry logic."""
        self._start_time = time.time()
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if not self.config.should_retry(e):
                    if self.config.on_give_up:
                        self.config.on_give_up(e)
                    raise
                
                # Check if this is the last attempt
                if attempt >= self.config.max_attempts:
                    if self.config.on_give_up:
                        self.config.on_give_up(e)
                    raise
                
                # Check time limit
                if not self._check_time_limit():
                    if self.config.on_give_up:
                        self.config.on_give_up(e)
                    raise
                
                # Calculate retry delay
                retry_after = self._extract_retry_after(e)
                delay = self.config.calculate_delay(attempt, retry_after)
                
                # Call retry callback if configured
                if self.config.on_retry:
                    self.config.on_retry(attempt, e, delay)
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # This shouldn't happen, but just in case
        if last_exception:
            raise last_exception
        raise APIError("Retry logic error: no exception captured")


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to a function."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                retry_manager = RetryManager(config)
                return await retry_manager.async_execute_with_retry(
                    lambda: func(*args, **kwargs)
                )
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                retry_manager = RetryManager(config)
                return retry_manager.execute_with_retry(
                    lambda: func(*args, **kwargs)
                )
            return sync_wrapper
    return decorator


# Preset retry configurations for common scenarios

CONSERVATIVE_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter=True,
)

AGGRESSIVE_RETRY = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    backoff_factor=2.5,
    jitter=True,
    max_retry_time=120.0,
)

NO_RETRY = RetryConfig(
    max_attempts=1,
)

RATE_LIMIT_RETRY = RetryConfig(
    max_attempts=5,
    initial_delay=5.0,
    max_delay=300.0,
    backoff_factor=2.0,
    jitter=True,
    respect_retry_after_header=True,
    retry_on_status_codes={429},
    retry_on_exceptions={APIRateLimitError},
)