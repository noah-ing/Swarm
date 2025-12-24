"""REST API client with comprehensive error handling for HTTP requests."""

import asyncio
import json
import time
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin

import httpx

from .config import APIConfig
from .retry import RetryManager
from .exceptions import (
    APIError,
    APITimeoutError,
    APIConnectionError,
    APIAuthenticationError,
    APIRateLimitError,
    APIValidationError,
    NetworkError,
    HTTPClientError,
    HTTPServerError,
    ResponseProcessingError,
    JSONDecodeError,
    ResponseEncodingError,
    get_error_from_status_code,
    classify_network_error,
)


class APIClient:
    """REST API client with comprehensive HTTP error handling and configuration."""

    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize API client with configuration.

        Args:
            config: API configuration. If None, uses default config.
        """
        self.config = config or APIConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
        self._retry_manager = RetryManager(self.config.retry_config)

    def _get_sync_client(self) -> httpx.Client:
        """Get or create synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout,
                    read=self.config.read_timeout,
                    write=self.config.read_timeout,  # Use read timeout for write
                    pool=self.config.total_timeout,
                ),
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_keepalive_connections,
                    keepalive_expiry=self.config.keepalive_expiry,
                ),
                headers=self.config.default_headers,
                base_url=self.config.base_url,
            )
        return self._sync_client

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create asynchronous HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout,
                    read=self.config.read_timeout,
                    write=self.config.read_timeout,  # Use read timeout for write
                    pool=self.config.total_timeout,
                ),
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_keepalive_connections,
                    keepalive_expiry=self.config.keepalive_expiry,
                ),
                headers=self.config.default_headers,
                base_url=self.config.base_url,
            )
        return self._client

    def _extract_rate_limit_info(self, response: httpx.Response) -> Dict[str, Any]:
        """Extract rate limit information from response headers."""
        rate_limit_info = {}

        # Standard headers
        if "X-RateLimit-Limit" in response.headers:
            try:
                rate_limit_info["limit"] = int(response.headers["X-RateLimit-Limit"])
            except ValueError:
                pass

        if "X-RateLimit-Remaining" in response.headers:
            try:
                rate_limit_info["remaining"] = int(response.headers["X-RateLimit-Remaining"])
            except ValueError:
                pass

        if "X-RateLimit-Reset" in response.headers:
            try:
                rate_limit_info["reset_time"] = int(response.headers["X-RateLimit-Reset"])
            except ValueError:
                pass

        if "Retry-After" in response.headers:
            try:
                rate_limit_info["retry_after"] = int(response.headers["Retry-After"])
            except ValueError:
                pass

        return rate_limit_info

    def _handle_response(
        self,
        response: httpx.Response,
        request_url: str,
        request_method: str
    ) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions with detailed context."""

        # Extract common error context
        error_context = {
            "request_url": request_url,
            "request_method": request_method,
            "headers": dict(response.headers),
            "status_code": response.status_code,
        }

        # Handle specific status codes with enhanced error information
        if response.status_code >= 400:
            try:
                error_data = response.json()
            except Exception:
                error_data = {"error": response.text}

            error_context["response"] = error_data

            # Rate limit errors with detailed information
            if response.status_code == 429:
                rate_limit_info = self._extract_rate_limit_info(response)
                raise APIRateLimitError(
                    "Rate limit exceeded",
                    response=response,
                    **rate_limit_info,
                    **error_context
                )

            # Authentication errors with specific context
            elif response.status_code == 401:
                auth_error_type = "unknown"
                error_message = "Authentication failed"

                if error_data:
                    if isinstance(error_data, dict):
                        if "token" in str(error_data).lower():
                            auth_error_type = "invalid_token"
                            error_message = "Invalid or expired authentication token"
                        elif "missing" in str(error_data).lower():
                            auth_error_type = "missing_token"
                            error_message = "Missing authentication credentials"

                raise APIAuthenticationError(
                    error_message,
                    auth_error_type=auth_error_type,
                    response=response,
                    **error_context
                )

            # Validation errors with field-specific information
            elif response.status_code == 422:
                validation_errors = {}
                if isinstance(error_data, dict) and "errors" in error_data:
                    validation_errors = error_data["errors"]
                elif isinstance(error_data, dict) and "detail" in error_data:
                    validation_errors = error_data["detail"]

                raise APIValidationError(
                    "Request validation failed",
                    validation_errors=validation_errors,
                    response=response,
                    **error_context
                )

            # Service unavailable with retry information
            elif response.status_code == 503:
                retry_after = None
                if "Retry-After" in response.headers:
                    try:
                        retry_after = int(response.headers["Retry-After"])
                    except ValueError:
                        pass

                raise get_error_from_status_code(
                    response.status_code,
                    "Service temporarily unavailable",
                    retry_after=retry_after,
                    response=response,
                    **error_context
                )

            # Use the factory function for other status codes
            else:
                error_message = f"API request failed with status {response.status_code}"
                if isinstance(error_data, dict) and "message" in error_data:
                    error_message = error_data["message"]
                elif isinstance(error_data, dict) and "error" in error_data:
                    error_message = error_data["error"]

                raise get_error_from_status_code(
                    response.status_code,
                    error_message,
                    response=response,
                    **error_context
                )

        # Parse successful response with error handling
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise JSONDecodeError(
                f"Failed to decode JSON response: {str(e)}",
                original_exception=e,
                **error_context
            )
        except UnicodeDecodeError as e:
            raise ResponseEncodingError(
                f"Response encoding error: {str(e)}",
                original_exception=e,
                **error_context
            )
        except Exception as e:
            # Fallback: return text response if JSON parsing fails for other reasons
            try:
                return {"data": response.text}
            except Exception:
                raise ResponseProcessingError(
                    f"Failed to process response: {str(e)}",
                    original_exception=e,
                    **error_context
                )

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if self.config.base_url:
            return urljoin(self.config.base_url, endpoint)
        return endpoint

    def _handle_network_exception(
        self,
        exception: Exception,
        request_url: str,
        request_method: str,
        retry_count: int = 0
    ) -> NetworkError:
        """Handle network exceptions with detailed classification."""

        error_context = {
            "request_url": request_url,
            "request_method": request_method,
            "retry_count": retry_count,
        }

        # Handle timeout exceptions
        if isinstance(exception, httpx.TimeoutException):
            timeout_type = "unknown"
            if isinstance(exception, httpx.ConnectTimeout):
                timeout_type = "connect"
            elif isinstance(exception, httpx.ReadTimeout):
                timeout_type = "read"
            elif isinstance(exception, httpx.PoolTimeout):
                timeout_type = "pool"
            elif isinstance(exception, httpx.WriteTimeout):
                timeout_type = "write"

            return APITimeoutError(
                f"Request timed out ({timeout_type}): {str(exception)}",
                timeout_type=timeout_type,
                original_exception=exception,
                **error_context
            )

        # Handle connection exceptions
        elif isinstance(exception, (httpx.ConnectError, httpx.NetworkError)):
            return classify_network_error(exception)

        # Handle other HTTPX exceptions
        elif isinstance(exception, httpx.HTTPError):
            return APIConnectionError(
                f"HTTP error: {str(exception)}",
                original_exception=exception,
                **error_context
            )

        # Generic network error
        return NetworkError(
            f"Network error: {str(exception)}",
            original_exception=exception,
            **error_context
        )

    # Synchronous methods

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send GET request with comprehensive error handling."""
        request_url = self._build_url(endpoint)

        def _request():
            client = self._get_sync_client()
            response = client.get(
                request_url,
                params=params,
                headers=headers,
            )
            return self._handle_response(response, request_url, "GET")

        return self._retry_manager.execute_with_retry(
            _request,
            context={"url": request_url, "method": "GET"}
        )

    def post(
        self,
        endpoint: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send POST request with comprehensive error handling."""
        request_url = self._build_url(endpoint)

        def _request():
            client = self._get_sync_client()

            # Prepare request body
            if isinstance(data, dict):
                json_data = data
                content = None
            else:
                json_data = None
                content = data

            response = client.post(
                request_url,
                json=json_data,
                content=content,
                params=params,
                headers=headers,
            )
            return self._handle_response(response, request_url, "POST")

        return self._retry_manager.execute_with_retry(
            _request,
            context={"url": request_url, "method": "POST"}
        )

    def put(
        self,
        endpoint: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send PUT request with comprehensive error handling."""
        request_url = self._build_url(endpoint)

        def _request():
            client = self._get_sync_client()

            # Prepare request body
            if isinstance(data, dict):
                json_data = data
                content = None
            else:
                json_data = None
                content = data

            response = client.put(
                request_url,
                json=json_data,
                content=content,
                params=params,
                headers=headers,
            )
            return self._handle_response(response, request_url, "PUT")

        return self._retry_manager.execute_with_retry(
            _request,
            context={"url": request_url, "method": "PUT"}
        )

    def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send DELETE request with comprehensive error handling."""
        request_url = self._build_url(endpoint)

        def _request():
            client = self._get_sync_client()
            response = client.delete(
                request_url,
                params=params,
                headers=headers,
            )
            return self._handle_response(response, request_url, "DELETE")

        return self._retry_manager.execute_with_retry(
            _request,
            context={"url": request_url, "method": "DELETE"}
        )

    # Asynchronous methods

    async def aget(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send async GET request with comprehensive error handling."""
        request_url = self._build_url(endpoint)

        async def _request():
            client = await self._get_async_client()
            response = await client.get(
                request_url,
                params=params,
                headers=headers,
            )
            return self._handle_response(response, request_url, "GET")

        return await self._retry_manager.async_execute_with_retry(
            _request,
            context={"url": request_url, "method": "GET"}
        )

    async def apost(
        self,
        endpoint: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send async POST request with comprehensive error handling."""
        request_url = self._build_url(endpoint)

        async def _request():
            client = await self._get_async_client()

            # Prepare request body
            if isinstance(data, dict):
                json_data = data
                content = None
            else:
                json_data = None
                content = data

            response = await client.post(
                request_url,
                json=json_data,
                content=content,
                params=params,
                headers=headers,
            )
            return self._handle_response(response, request_url, "POST")

        return await self._retry_manager.async_execute_with_retry(
            _request,
            context={"url": request_url, "method": "POST"}
        )

    async def aput(
        self,
        endpoint: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send async PUT request with comprehensive error handling."""
        request_url = self._build_url(endpoint)

        async def _request():
            client = await self._get_async_client()

            # Prepare request body
            if isinstance(data, dict):
                json_data = data
                content = None
            else:
                json_data = None
                content = data

            response = await client.put(
                request_url,
                json=json_data,
                content=content,
                params=params,
                headers=headers,
            )
            return self._handle_response(response, request_url, "PUT")

        return await self._retry_manager.async_execute_with_retry(
            _request,
            context={"url": request_url, "method": "PUT"}
        )

    async def adelete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send async DELETE request with comprehensive error handling."""
        request_url = self._build_url(endpoint)

        async def _request():
            client = await self._get_async_client()
            response = await client.delete(
                request_url,
                params=params,
                headers=headers,
            )
            return self._handle_response(response, request_url, "DELETE")

        return await self._retry_manager.async_execute_with_retry(
            _request,
            context={"url": request_url, "method": "DELETE"}
        )

    def close(self):
        """Close HTTP clients."""
        if self._sync_client:
            self._sync_client.close()
        if self._client:
            asyncio.create_task(self._client.aclose())

    async def aclose(self):
        """Close async HTTP client."""
        if self._client:
            await self._client.aclose()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()