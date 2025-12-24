"""Comprehensive tests for API client error handling."""

import json
import pytest
from unittest.mock import Mock, patch
import httpx

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
    BadGatewayError,
    ServiceUnavailableError,
    HTTPClientError,
    HTTPServerError,
    JSONDecodeError,
    ResponseEncodingError,
    get_error_from_status_code,
    classify_network_error,
    DNSResolutionError,
    ConnectionRefusedError,
    SSLError,
)


class TestExceptionHierarchy:
    """Test the exception class hierarchy and basic functionality."""

    def test_base_api_error(self):
        """Test base APIError class functionality."""
        error = APIError(
            "Test error",
            status_code=500,
            response={"error": "test"},
            request_url="https://api.example.com/test",
            request_method="GET",
            retry_count=2
        )
        
        assert str(error) == "Test error | Status: 500 | Request: GET https://api.example.com/test | Retries: 2"
        assert error.status_code == 500
        assert error.response == {"error": "test"}
        assert error.retry_count == 2
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "APIError"
        assert error_dict["message"] == "Test error"
        assert error_dict["status_code"] == 500

    def test_specific_error_classes(self):
        """Test specific error classes have correct status codes."""
        assert BadRequestError().status_code == 400
        assert APIAuthenticationError().status_code == 401
        assert NotFoundError().status_code == 404
        assert APIValidationError().status_code == 422
        assert APIRateLimitError().status_code == 429
        assert InternalServerError().status_code == 500
        assert ServiceUnavailableError().status_code == 503

    def test_rate_limit_error_with_info(self):
        """Test rate limit error with additional information."""
        error = APIRateLimitError(
            retry_after=30,
            limit=1000,
            remaining=0,
            reset_time=1234567890
        )
        
        assert error.retry_after == 30
        assert error.limit == 1000
        assert error.remaining == 0
        assert error.reset_time == 1234567890
        assert "Retry after: 30s" in str(error)

    def test_authentication_error_types(self):
        """Test authentication error with type classification."""
        auth_error = APIAuthenticationError(
            "Invalid token",
            auth_error_type="invalid_token"
        )
        
        assert auth_error.auth_error_type == "invalid_token"

    def test_validation_error_with_details(self):
        """Test validation error with field-specific errors."""
        validation_errors = {
            "email": ["Invalid email format"],
            "password": ["Password too short"]
        }
        
        error = APIValidationError(
            "Validation failed",
            validation_errors=validation_errors
        )
        
        assert error.validation_errors == validation_errors


class TestErrorClassification:
    """Test error classification utility functions."""

    def test_get_error_from_status_code_4xx(self):
        """Test 4xx error classification."""
        # Test specific 4xx codes
        assert isinstance(get_error_from_status_code(400), BadRequestError)
        assert isinstance(get_error_from_status_code(401), APIAuthenticationError)
        assert isinstance(get_error_from_status_code(404), NotFoundError)
        assert isinstance(get_error_from_status_code(422), APIValidationError)
        assert isinstance(get_error_from_status_code(429), APIRateLimitError)
        
        # Test generic 4xx
        error = get_error_from_status_code(418, "I'm a teapot")
        assert isinstance(error, HTTPClientError)
        assert error.status_code == 418
        assert "I'm a teapot" in str(error)

    def test_get_error_from_status_code_5xx(self):
        """Test 5xx error classification."""
        assert isinstance(get_error_from_status_code(500), InternalServerError)
        assert isinstance(get_error_from_status_code(502), BadGatewayError)
        assert isinstance(get_error_from_status_code(503), ServiceUnavailableError)
        
        # Test generic 5xx
        error = get_error_from_status_code(599, "Custom server error")
        assert isinstance(error, HTTPServerError)
        assert error.status_code == 599

    def test_classify_network_error_dns(self):
        """Test DNS error classification."""
        dns_exception = Exception("DNS resolution failed")
        error = classify_network_error(dns_exception)
        assert isinstance(error, DNSResolutionError)
        assert error.connection_error_type == "dns"

    def test_classify_network_error_connection_refused(self):
        """Test connection refused error classification."""
        refused_exception = Exception("Connection refused")
        error = classify_network_error(refused_exception)
        assert isinstance(error, ConnectionRefusedError)
        assert error.connection_error_type == "refused"

    def test_classify_network_error_ssl(self):
        """Test SSL error classification."""
        ssl_exception = Exception("SSL handshake failed")
        error = classify_network_error(ssl_exception)
        assert isinstance(error, SSLError)


class TestClientErrorHandling:
    """Test error handling in the APIClient."""

    def setup_method(self):
        """Set up test client."""
        self.config = APIConfig(base_url="https://api.example.com")
        self.client = APIClient(self.config)

    def test_timeout_error_handling(self):
        """Test timeout error handling."""
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get.side_effect = httpx.ConnectTimeout("Connection timed out")
            
            with pytest.raises(APITimeoutError) as exc_info:
                self.client.get("/test")
            
            assert exc_info.value.timeout_type == "connect"
            assert "Connection timed out" in str(exc_info.value)

    def test_connection_error_handling(self):
        """Test connection error handling."""
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            
            with pytest.raises(APIConnectionError) as exc_info:
                self.client.get("/test")
            
            assert "Connection failed" in str(exc_info.value)

    def test_http_error_response_handling(self):
        """Test HTTP error response handling."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Resource not found"}
        mock_response.headers = {}
        
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get.return_value = mock_response
            
            with pytest.raises(NotFoundError) as exc_info:
                self.client.get("/test")
            
            assert exc_info.value.status_code == 404
            assert exc_info.value.response == {"error": "Resource not found"}

    def test_rate_limit_header_parsing(self):
        """Test rate limit header parsing."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_response.headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1234567890",
            "Retry-After": "30"
        }
        
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get.return_value = mock_response
            
            with pytest.raises(APIRateLimitError) as exc_info:
                self.client.get("/test")
            
            error = exc_info.value
            assert error.limit == 1000
            assert error.remaining == 0
            assert error.reset_time == 1234567890
            assert error.retry_after == 30

    def test_authentication_error_classification(self):
        """Test authentication error type classification."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid token provided"}
        mock_response.headers = {}
        
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get.return_value = mock_response
            
            with pytest.raises(APIAuthenticationError) as exc_info:
                self.client.get("/test")
            
            error = exc_info.value
            assert error.auth_error_type == "invalid_token"
            assert "Invalid or expired authentication token" in str(error)

    def test_validation_error_details(self):
        """Test validation error details extraction."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "errors": {
                "email": ["Invalid format"],
                "age": ["Must be positive"]
            }
        }
        mock_response.headers = {}
        
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            with pytest.raises(APIValidationError) as exc_info:
                self.client.post("/test", data={"email": "invalid", "age": -1})
            
            error = exc_info.value
            assert "email" in error.validation_errors
            assert "age" in error.validation_errors

    def test_json_decode_error_handling(self):
        """Test JSON decode error handling."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        mock_response.text = "Invalid JSON response"
        mock_response.headers = {}
        
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get.return_value = mock_response
            
            with pytest.raises(JSONDecodeError) as exc_info:
                self.client.get("/test")
            
            assert "Failed to decode JSON response" in str(exc_info.value)

    def test_service_unavailable_with_retry_after(self):
        """Test service unavailable error with retry-after header."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.json.return_value = {"error": "Service temporarily unavailable"}
        mock_response.headers = {"Retry-After": "120"}
        
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get.return_value = mock_response
            
            with pytest.raises(ServiceUnavailableError) as exc_info:
                self.client.get("/test")
            
            assert exc_info.value.retry_after == 120

    def test_retry_logic_with_network_errors(self):
        """Test retry logic with network errors."""
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # First two calls fail, third succeeds
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.headers = {}
            
            mock_client.get.side_effect = [
                httpx.ConnectTimeout("Timeout 1"),
                httpx.ConnectTimeout("Timeout 2"),
                mock_response
            ]
            
            with patch('time.sleep'):  # Mock sleep to speed up test
                result = self.client.get("/test")
            
            assert result == {"success": True}
            assert mock_client.get.call_count == 3

    def test_no_retry_for_client_errors(self):
        """Test that client errors (4xx) are not retried."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_response.headers = {}
        
        with patch.object(self.client, '_get_sync_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get.return_value = mock_response
            
            with pytest.raises(BadRequestError):
                self.client.get("/test")
            
            # Should only be called once (no retries)
            assert mock_client.get.call_count == 1


class TestAsyncErrorHandling:
    """Test async error handling functionality."""

    def setup_method(self):
        """Set up test client."""
        self.config = APIConfig(base_url="https://api.example.com")
        self.client = APIClient(self.config)

    def test_async_timeout_error(self):
        """Test async timeout error handling."""
        import asyncio
        
        async def run_test():
            with patch.object(self.client, '_get_async_client') as mock_get_client:
                mock_client = Mock()
                mock_get_client.return_value = mock_client
                mock_client.get.side_effect = httpx.ReadTimeout("Read timed out")
                
                with pytest.raises(APITimeoutError) as exc_info:
                    await self.client.aget("/test")
                
                assert exc_info.value.timeout_type == "read"
        
        asyncio.run(run_test())

    def test_async_retry_logic(self):
        """Test async retry logic."""
        import asyncio
        
        async def run_test():
            with patch.object(self.client, '_get_async_client') as mock_get_client:
                mock_client = Mock()
                mock_get_client.return_value = mock_client
                
                # First call fails, second succeeds
                mock_response = Mock(spec=httpx.Response)
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True}
                mock_response.headers = {}
                
                mock_client.get.side_effect = [
                    httpx.ConnectError("Connection failed"),
                    mock_response
                ]
                
                with patch('asyncio.sleep'):  # Mock sleep to speed up test
                    result = await self.client.aget("/test")
                
                assert result == {"success": True}
                assert mock_client.get.call_count == 2
        
        asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])