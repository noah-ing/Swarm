# API Client with Comprehensive Error Handling

A robust REST API client library with comprehensive error handling for different HTTP error types (4xx, 5xx), network errors, timeouts, and response processing errors.

## Features

- 🚀 **Sync & Async Support**: Full support for both synchronous and asynchronous operations
- 🛡️ **Comprehensive Error Handling**: Detailed exception hierarchy for all error types
- 🔄 **Smart Retry Logic**: Configurable retry with exponential backoff
- ⚙️ **Flexible Configuration**: Extensive timeout, connection, and retry settings
- 📊 **Rich Error Context**: Detailed error information for debugging and logging
- 🌐 **Network Error Classification**: Automatic classification of network issues
- 📈 **Rate Limit Handling**: Built-in rate limit detection and retry-after support

## Installation

```bash
pip install httpx  # Required dependency
```

## Quick Start

### Basic Usage

```python
from api_client import APIClient, APIConfig

# Create client with default configuration
client = APIClient()

# Or with custom configuration
config = APIConfig(
    base_url="https://api.example.com",
    max_retries=3,
    retry_delay=1.0,
    connection_timeout=10.0
)
client = APIClient(config)

# Make requests
try:
    response = client.get("/users")
    print(response)
except Exception as e:
    print(f"Request failed: {e}")
```

### Async Usage

```python
import asyncio
from api_client import APIClient

async def main():
    async with APIClient() as client:
        try:
            response = await client.aget("/users")
            print(response)
        except Exception as e:
            print(f"Request failed: {e}")

asyncio.run(main())
```

## Error Handling

### Exception Hierarchy

```
APIError (base)
├── NetworkError
│   ├── APITimeoutError
│   ├── APIConnectionError
│   │   ├── DNSResolutionError
│   │   ├── ConnectionRefusedError
│   │   └── NetworkUnreachableError
│   └── SSLError
├── HTTPClientError (4xx)
│   ├── BadRequestError (400)
│   ├── APIAuthenticationError (401)
│   ├── ForbiddenError (403)
│   ├── NotFoundError (404)
│   ├── MethodNotAllowedError (405)
│   ├── ConflictError (409)
│   ├── APIValidationError (422)
│   └── APIRateLimitError (429)
├── HTTPServerError (5xx)
│   ├── InternalServerError (500)
│   ├── BadGatewayError (502)
│   ├── ServiceUnavailableError (503)
│   └── GatewayTimeoutError (504)
├── ResponseProcessingError
│   ├── JSONDecodeError
│   └── ResponseEncodingError
└── ConfigurationError
    ├── MissingConfigurationError
    └── InvalidConfigurationError
```

### Basic Error Handling

```python
from api_client import (
    APIClient,
    NotFoundError,
    APIAuthenticationError,
    APIRateLimitError,
    APITimeoutError,
    HTTPServerError
)

client = APIClient()

try:
    response = client.get("/api/data")
except NotFoundError as e:
    print(f"Resource not found: {e}")
except APIAuthenticationError as e:
    print(f"Authentication failed: {e}")
    print(f"Auth error type: {e.auth_error_type}")
except APIRateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    if e.retry_after:
        print(f"Retry after {e.retry_after} seconds")
except APITimeoutError as e:
    print(f"Request timed out: {e.timeout_type}")
except HTTPServerError as e:
    print(f"Server error: {e.status_code}")
```

### Advanced Error Handling

```python
from api_client import APIError, HTTPClientError, NetworkError

try:
    response = client.post("/api/users", data=user_data)
except HTTPClientError as e:
    # Handle all 4xx errors
    print(f"Client error: {e.status_code} - {e.message}")
    if hasattr(e, 'validation_errors'):
        print(f"Validation errors: {e.validation_errors}")
except NetworkError as e:
    # Handle all network-related errors
    print(f"Network error: {e}")
    print(f"Retry count: {e.retry_count}")
except APIError as e:
    # Handle any other API errors
    print(f"API error: {e}")
    
    # Get comprehensive error information
    error_info = e.to_dict()
    print(f"Full error context: {error_info}")
```

### Rate Limit Handling

```python
from api_client import APIRateLimitError
import time

try:
    response = client.get("/api/data")
except APIRateLimitError as e:
    print(f"Rate limit exceeded!")
    print(f"Limit: {e.limit}")
    print(f"Remaining: {e.remaining}")
    print(f"Reset time: {e.reset_time}")
    
    if e.retry_after:
        print(f"Waiting {e.retry_after} seconds...")
        time.sleep(e.retry_after)
        # Retry the request
        response = client.get("/api/data")
```

### Network Error Classification

```python
from api_client import (
    DNSResolutionError,
    ConnectionRefusedError,
    NetworkUnreachableError,
    SSLError
)

try:
    response = client.get("/api/data")
except DNSResolutionError as e:
    print("DNS resolution failed - check domain name")
except ConnectionRefusedError as e:
    print("Connection refused - server may be down")
except NetworkUnreachableError as e:
    print("Network unreachable - check connectivity")
except SSLError as e:
    print("SSL/TLS error - check certificates")
```

## Configuration

### APIConfig Options

```python
from api_client import APIConfig

config = APIConfig(
    # Base URL for all requests
    base_url="https://api.example.com",
    
    # Timeout settings (seconds)
    connection_timeout=10.0,
    read_timeout=30.0,
    total_timeout=60.0,
    
    # Connection pool settings
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=30.0,
    
    # Retry settings
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    
    # Default headers
    default_headers={
        "User-Agent": "MyApp/1.0",
        "Accept": "application/json",
        "Authorization": "Bearer your-token"
    }
)

client = APIClient(config)
```

### Environment Variable Configuration

```python
import os
from api_client import APIConfig

config = APIConfig(
    base_url=os.getenv("API_BASE_URL", "https://api.example.com"),
    connection_timeout=float(os.getenv("API_TIMEOUT", "10")),
    max_retries=int(os.getenv("API_MAX_RETRIES", "3")),
    default_headers={
        "Authorization": f"Bearer {os.getenv('API_TOKEN')}"
    }
)
```

## Advanced Usage

### Custom Error Handling

```python
def robust_api_call(client, endpoint, max_attempts=3):
    """Make API call with sophisticated error handling."""
    for attempt in range(max_attempts):
        try:
            return client.get(endpoint)
        
        except APIRateLimitError as e:
            if e.retry_after and attempt < max_attempts - 1:
                time.sleep(e.retry_after)
                continue
            raise
        
        except APITimeoutError as e:
            if attempt < max_attempts - 1:
                print(f"Timeout ({e.timeout_type}), retrying...")
                continue
            raise
        
        except HTTPServerError as e:
            if e.status_code == 503 and attempt < max_attempts - 1:
                print("Service unavailable, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
        
        except HTTPClientError:
            # Don't retry client errors
            raise
    
    raise APIError("Max attempts exceeded")

# Usage
try:
    result = robust_api_call(client, "/api/data")
except Exception as e:
    print(f"Failed after retries: {e}")
```

### Logging Integration

```python
import logging
from api_client import APIError

# Set up structured logging
logger = logging.getLogger(__name__)

try:
    response = client.get("/api/data")
except APIError as e:
    # Log with structured data
    logger.error("API request failed", extra={
        "error_type": type(e).__name__,
        "status_code": getattr(e, 'status_code', None),
        "request_url": getattr(e, 'request_url', None),
        "retry_count": getattr(e, 'retry_count', 0),
        "error_details": e.to_dict()
    })
```

### Context Managers

```python
# Synchronous context manager
with APIClient(config) as client:
    response = client.get("/api/data")
# Client is automatically closed

# Asynchronous context manager
async with APIClient(config) as client:
    response = await client.aget("/api/data")
# Client is automatically closed
```

## Error Information

### Comprehensive Error Context

Every exception includes detailed context information:

```python
try:
    response = client.post("/api/users", data={"name": "test"})
except APIError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Message: {e.message}")
    print(f"Status code: {e.status_code}")
    print(f"Request URL: {e.request_url}")
    print(f"Request method: {e.request_method}")
    print(f"Retry count: {e.retry_count}")
    print(f"Timestamp: {e.timestamp}")
    print(f"Response data: {e.response}")
    print(f"Headers: {e.headers}")
    
    # Get all information as dictionary
    error_dict = e.to_dict()
    print(f"Complete error info: {error_dict}")
```

### Utility Functions

```python
from api_client import get_error_from_status_code, classify_network_error

# Get appropriate exception for status code
error = get_error_from_status_code(404, "Custom not found message")
print(type(error))  # <class 'NotFoundError'>

# Classify network exceptions
import socket
try:
    # Some network operation
    pass
except socket.gaierror as e:
    classified_error = classify_network_error(e)
    print(type(classified_error))  # <class 'DNSResolutionError'>
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest api_client/test_error_handling.py -v
```

## Examples

See `example_error_handling.py` for comprehensive examples of:
- Basic error handling patterns
- Network error handling
- Server error handling
- Error hierarchy usage
- Async error handling
- Custom error handling patterns
- Error logging integration

## License

MIT License - see LICENSE file for details.