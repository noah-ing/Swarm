"""Specialized API client exceptions."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BaseAPIError(Exception):
    """Base exception for all API-related errors."""
    
    def __init__(
        self,
        message: str,
        original_exception: Optional[Exception] = None,
        **context
    ):
        """
        Initialize API error with context and logging.
        
        Args:
            message (str): Primary error message
            original_exception (Optional[Exception]): Original exception
            **context: Additional error context
        """
        self.message = message
        self.original_exception = original_exception
        self.context = context
        
        # Enhanced logging
        log_message = f"{self.__class__.__name__}: {message}"
        if context:
            log_message += f" | Context: {context}"
        
        logger.error(log_message)
        
        super().__init__(message)

class APIConnectionError(BaseAPIError):
    """Raised for general connection-related API errors."""
    pass

class APITimeoutError(BaseAPIError):
    """Raised when an API request times out."""
    def __init__(self, message: str, timeout_type: str = "unknown", **kwargs):
        self.timeout_type = timeout_type
        super().__init__(message, **kwargs)

class APIRateLimitError(BaseAPIError):
    """Raised when API rate limit is exceeded."""
    def __init__(
        self, 
        message: str, 
        remaining: Optional[int] = None, 
        limit: Optional[int] = None, 
        reset_time: Optional[int] = None,
        **kwargs
    ):
        self.remaining = remaining
        self.limit = limit
        self.reset_time = reset_time
        super().__init__(message, **kwargs)

class APIValidationError(BaseAPIError):
    """Raised when request validation fails."""
    def __init__(
        self, 
        message: str, 
        validation_errors: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.validation_errors = validation_errors or {}
        super().__init__(message, **kwargs)

class JSONDecodeError(BaseAPIError):
    """Raised when JSON decoding fails."""
    pass

class ResponseEncodingError(BaseAPIError):
    """Raised for response encoding issues."""
    pass

class ResponseProcessingError(BaseAPIError):
    """Raised when generic response processing fails."""
    pass

def get_error_from_status_code(
    status_code: int, 
    message: str, 
    **kwargs
) -> BaseAPIError:
    """
    Map HTTP status codes to appropriate exceptions.
    
    Args:
        status_code (int): HTTP status code
        message (str): Error message
        **kwargs: Additional context
    
    Returns:
        BaseAPIError: Appropriate exception for the status code
    """
    error_map = {
        400: APIConnectionError,
        401: APIConnectionError,
        403: APIConnectionError,
        404: APIConnectionError,
        429: APIRateLimitError,
        500: APIConnectionError,
        502: APIConnectionError,
        503: APIConnectionError,
        504: APITimeoutError,
    }
    
    ExceptionClass = error_map.get(status_code, BaseAPIError)
    return ExceptionClass(f"{status_code} - {message}", **kwargs)

def classify_network_error(exception: Exception) -> APIConnectionError:
    """
    Classify and create an appropriate network error.
    
    Args:
        exception (Exception): Original network exception
    
    Returns:
        APIConnectionError: Classified connection error
    """
    error_types = {
        "connection_error": "Unable to connect to the server",
        "dns_resolution_error": "Failed to resolve server address",
        "network_timeout": "Network request timed out",
    }
    
    for error_type, error_message in error_types.items():
        if error_type in str(exception).lower():
            return APIConnectionError(
                error_message, 
                original_exception=exception
            )
    
    return APIConnectionError(
        f"Network error: {str(exception)}", 
        original_exception=exception
    )