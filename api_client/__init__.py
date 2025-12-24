"""REST API client with comprehensive error handling and retry logic."""

from .client import APIClient
from .config import APIConfig
from .retry import (
    RetryConfig,
    RetryManager,
    with_retry,
    CONSERVATIVE_RETRY,
    AGGRESSIVE_RETRY,
    NO_RETRY,
    RATE_LIMIT_RETRY,
)
from .exceptions import (
    # Base exceptions
    APIError,
    NetworkError,
    ResponseError,
    
    # Network exceptions
    APIConnectionError,
    APITimeoutError,
    DNSResolutionError,
    SSLError,
    ConnectionRefusedError,
    ConnectionResetError,
    HostUnreachableError,
    NetworkUnreachableError,
    ProxyError,
    
    # HTTP Client errors (4xx)
    HTTPClientError,
    BadRequestError,
    APIAuthenticationError,
    ForbiddenError,
    NotFoundError,
    MethodNotAllowedError,
    NotAcceptableError,
    ProxyAuthenticationRequiredError,
    RequestTimeoutError,
    ConflictError,
    GoneError,
    LengthRequiredError,
    PreconditionFailedError,
    PayloadTooLargeError,
    URITooLongError,
    UnsupportedMediaTypeError,
    RequestedRangeNotSatisfiableError,
    ExpectationFailedError,
    APIValidationError,
    LockedError,
    FailedDependencyError,
    TooEarlyError,
    UpgradeRequiredError,
    PreconditionRequiredError,
    APIRateLimitError,
    RequestHeaderFieldsTooLargeError,
    UnavailableForLegalReasonsError,
    
    # HTTP Server errors (5xx)
    HTTPServerError,
    InternalServerError,
    NotImplementedError,
    BadGatewayError,
    ServiceUnavailableError,
    GatewayTimeoutError,
    HTTPVersionNotSupportedError,
    VariantAlsoNegotiatesError,
    InsufficientStorageError,
    LoopDetectedError,
    NotExtendedError,
    NetworkAuthenticationRequiredError,
    
    # Response processing errors
    ResponseProcessingError,
    JSONDecodeError,
    ResponseEncodingError,
    
    # Utility functions
    get_error_from_status_code,
    classify_network_error,
)

__all__ = [
    # Client
    "APIClient",
    "APIConfig",
    
    # Retry
    "RetryConfig",
    "RetryManager",
    "with_retry",
    "CONSERVATIVE_RETRY",
    "AGGRESSIVE_RETRY",
    "NO_RETRY",
    "RATE_LIMIT_RETRY",
    
    # Base exceptions
    "APIError",
    "NetworkError",
    "ResponseError",
    
    # Network exceptions
    "APIConnectionError",
    "APITimeoutError",
    "DNSResolutionError",
    "SSLError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "HostUnreachableError",
    "NetworkUnreachableError",
    "ProxyError",
    
    # HTTP Client errors
    "HTTPClientError",
    "BadRequestError",
    "APIAuthenticationError",
    "ForbiddenError",
    "NotFoundError",
    "MethodNotAllowedError",
    "NotAcceptableError",
    "ProxyAuthenticationRequiredError",
    "RequestTimeoutError",
    "ConflictError",
    "GoneError",
    "LengthRequiredError",
    "PreconditionFailedError",
    "PayloadTooLargeError",
    "URITooLongError",
    "UnsupportedMediaTypeError",
    "RequestedRangeNotSatisfiableError",
    "ExpectationFailedError",
    "APIValidationError",
    "LockedError",
    "FailedDependencyError",
    "TooEarlyError",
    "UpgradeRequiredError",
    "PreconditionRequiredError",
    "APIRateLimitError",
    "RequestHeaderFieldsTooLargeError",
    "UnavailableForLegalReasonsError",
    
    # HTTP Server errors
    "HTTPServerError",
    "InternalServerError",
    "NotImplementedError",
    "BadGatewayError",
    "ServiceUnavailableError",
    "GatewayTimeoutError",
    "HTTPVersionNotSupportedError",
    "VariantAlsoNegotiatesError",
    "InsufficientStorageError",
    "LoopDetectedError",
    "NotExtendedError",
    "NetworkAuthenticationRequiredError",
    
    # Response processing errors
    "ResponseProcessingError",
    "JSONDecodeError",
    "ResponseEncodingError",
    
    # Utility functions
    "get_error_from_status_code",
    "classify_network_error",
]