"""Examples demonstrating comprehensive API client error handling."""

import asyncio
import json
from api_client import (
    APIClient,
    APIConfig,
    APIError,
    APITimeoutError,
    APIConnectionError,
    APIAuthenticationError,
    APIRateLimitError,
    APIValidationError,
    BadRequestError,
    NotFoundError,
    InternalServerError,
    ServiceUnavailableError,
    JSONDecodeError,
    NetworkError,
    HTTPClientError,
    HTTPServerError,
)


def demonstrate_basic_error_handling():
    """Demonstrate basic error handling patterns."""
    print("=== Basic Error Handling Examples ===\n")
    
    config = APIConfig(
        base_url="https://httpbin.org",
        max_retries=2,
        retry_delay=1.0
    )
    
    client = APIClient(config)
    
    # Example 1: Handle 404 Not Found
    try:
        response = client.get("/status/404")
    except NotFoundError as e:
        print(f"Resource not found: {e}")
        print(f"Status code: {e.status_code}")
        print(f"Request URL: {e.request_url}")
        print()
    
    # Example 2: Handle authentication error
    try:
        response = client.get("/status/401")
    except APIAuthenticationError as e:
        print(f"Authentication failed: {e}")
        print(f"Auth error type: {e.auth_error_type}")
        print(f"Error details: {e.to_dict()}")
        print()
    
    # Example 3: Handle rate limiting
    try:
        response = client.get("/status/429")
    except APIRateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        if hasattr(e, 'retry_after') and e.retry_after:
            print(f"Retry after: {e.retry_after} seconds")
        print()
    
    # Example 4: Handle validation errors
    try:
        response = client.post("/status/422", data={"invalid": "data"})
    except APIValidationError as e:
        print(f"Validation failed: {e}")
        print(f"Validation errors: {e.validation_errors}")
        print()


def demonstrate_network_error_handling():
    """Demonstrate network error handling."""
    print("=== Network Error Handling Examples ===\n")
    
    # Configure client with shorter timeouts for demonstration
    config = APIConfig(
        base_url="https://httpbin.org",
        connection_timeout=0.1,  # Very short timeout
        read_timeout=0.1,
        max_retries=1
    )
    
    client = APIClient(config)
    
    # Example 1: Handle timeout errors
    try:
        response = client.get("/delay/5")  # This will timeout
    except APITimeoutError as e:
        print(f"Request timed out: {e}")
        print(f"Timeout type: {e.timeout_type}")
        print(f"Retry count: {e.retry_count}")
        print()
    except Exception as e:
        print(f"Other error: {e}")
        print()
    
    # Example 2: Handle connection errors
    config.base_url = "https://nonexistent-domain-12345.com"
    client = APIClient(config)
    
    try:
        response = client.get("/test")
    except APIConnectionError as e:
        print(f"Connection failed: {e}")
        print(f"Connection error type: {e.connection_error_type}")
        print()
    except NetworkError as e:
        print(f"Network error: {e}")
        print()


def demonstrate_server_error_handling():
    """Demonstrate server error handling."""
    print("=== Server Error Handling Examples ===\n")
    
    config = APIConfig(base_url="https://httpbin.org")
    client = APIClient(config)
    
    # Example 1: Handle internal server error
    try:
        response = client.get("/status/500")
    except InternalServerError as e:
        print(f"Internal server error: {e}")
        print(f"Status: {e.status_code}")
        print()
    
    # Example 2: Handle service unavailable
    try:
        response = client.get("/status/503")
    except ServiceUnavailableError as e:
        print(f"Service unavailable: {e}")
        if hasattr(e, 'retry_after') and e.retry_after:
            print(f"Retry after: {e.retry_after} seconds")
        print()


def demonstrate_error_hierarchy_handling():
    """Demonstrate handling errors using the exception hierarchy."""
    print("=== Error Hierarchy Handling Examples ===\n")
    
    config = APIConfig(base_url="https://httpbin.org")
    client = APIClient(config)
    
    endpoints = ["/status/400", "/status/401", "/status/404", "/status/500", "/status/502"]
    
    for endpoint in endpoints:
        try:
            response = client.get(endpoint)
            print(f"Success: {endpoint}")
        except HTTPClientError as e:
            print(f"Client error ({endpoint}): {e}")
            print(f"  - Type: {type(e).__name__}")
            print(f"  - Status: {e.status_code}")
        except HTTPServerError as e:
            print(f"Server error ({endpoint}): {e}")
            print(f"  - Type: {type(e).__name__}")
            print(f"  - Status: {e.status_code}")
        except APIError as e:
            print(f"General API error ({endpoint}): {e}")
        print()


def demonstrate_comprehensive_error_context():
    """Demonstrate comprehensive error context information."""
    print("=== Comprehensive Error Context Examples ===\n")
    
    config = APIConfig(base_url="https://httpbin.org")
    client = APIClient(config)
    
    try:
        response = client.post("/status/422", data={"test": "data"})
    except APIError as e:
        print("Complete error information:")
        error_dict = e.to_dict()
        for key, value in error_dict.items():
            print(f"  {key}: {value}")
        print()


async def demonstrate_async_error_handling():
    """Demonstrate async error handling."""
    print("=== Async Error Handling Examples ===\n")
    
    config = APIConfig(base_url="https://httpbin.org", max_retries=1)
    
    async with APIClient(config) as client:
        # Example 1: Async 404 error
        try:
            response = await client.aget("/status/404")
        except NotFoundError as e:
            print(f"Async 404 error: {e}")
            print()
        
        # Example 2: Async timeout with very short timeout
        config.connection_timeout = 0.01
        timeout_client = APIClient(config)
        
        try:
            response = await timeout_client.aget("/delay/1")
        except APITimeoutError as e:
            print(f"Async timeout: {e}")
            print(f"Timeout type: {e.timeout_type}")
            print()
        except Exception as e:
            print(f"Other async error: {e}")
            print()
        finally:
            await timeout_client.aclose()


def demonstrate_custom_error_handling():
    """Demonstrate custom error handling patterns."""
    print("=== Custom Error Handling Patterns ===\n")
    
    config = APIConfig(base_url="https://httpbin.org")
    client = APIClient(config)
    
    def handle_api_request(client, endpoint, max_attempts=3):
        """Custom function with sophisticated error handling."""
        for attempt in range(max_attempts):
            try:
                return client.get(endpoint)
            
            except APIRateLimitError as e:
                print(f"Rate limited on attempt {attempt + 1}")
                if e.retry_after:
                    print(f"Waiting {e.retry_after} seconds...")
                    # In real code, you'd sleep here
                if attempt < max_attempts - 1:
                    continue
                raise
            
            except APITimeoutError as e:
                print(f"Timeout on attempt {attempt + 1}: {e.timeout_type}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                    continue
                raise
            
            except HTTPClientError as e:
                print(f"Client error (no retry): {e}")
                raise
            
            except HTTPServerError as e:
                print(f"Server error on attempt {attempt + 1}: {e}")
                if e.status_code == 503:  # Service unavailable - retry
                    if attempt < max_attempts - 1:
                        print("Service unavailable, retrying...")
                        continue
                elif e.status_code >= 500:  # Other server errors - retry
                    if attempt < max_attempts - 1:
                        print("Server error, retrying...")
                        continue
                raise
            
            except NetworkError as e:
                print(f"Network error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    print("Network error, retrying...")
                    continue
                raise
        
        raise APIError("Max attempts exceeded")
    
    # Test the custom handler
    try:
        result = handle_api_request(client, "/status/503")
    except Exception as e:
        print(f"Final error: {e}")
        print()


def demonstrate_error_logging():
    """Demonstrate error logging patterns."""
    print("=== Error Logging Examples ===\n")
    
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    config = APIConfig(base_url="https://httpbin.org")
    client = APIClient(config)
    
    try:
        response = client.get("/status/500")
    except APIError as e:
        # Log structured error information
        logger.error("API request failed", extra={
            "error_type": type(e).__name__,
            "status_code": getattr(e, 'status_code', None),
            "request_url": getattr(e, 'request_url', None),
            "request_method": getattr(e, 'request_method', None),
            "retry_count": getattr(e, 'retry_count', 0),
            "error_dict": e.to_dict()
        })
        print("Error logged with structured data")
        print()


if __name__ == "__main__":
    print("API Client Error Handling Examples")
    print("=" * 40)
    print()
    
    # Run synchronous examples
    demonstrate_basic_error_handling()
    demonstrate_network_error_handling() 
    demonstrate_server_error_handling()
    demonstrate_error_hierarchy_handling()
    demonstrate_comprehensive_error_context()
    demonstrate_custom_error_handling()
    demonstrate_error_logging()
    
    # Run async examples
    print("Running async examples...")
    asyncio.run(demonstrate_async_error_handling())
    
    print("All examples completed!")