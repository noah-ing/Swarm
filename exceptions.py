"""
Centralized exception handling for the application.
Provides custom exceptions with descriptive messages.
"""

class SwarmBaseException(Exception):
    """Base exception for all Swarm-related errors."""
    def __init__(self, message="An unexpected error occurred in Swarm", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class ConfigurationError(SwarmBaseException):
    """Raised when there's an issue with configuration."""
    pass

class AuthenticationError(SwarmBaseException):
    """Raised when authentication fails."""
    pass

class ResourceNotFoundError(SwarmBaseException):
    """Raised when a requested resource cannot be found."""
    pass

class PermissionDeniedError(SwarmBaseException):
    """Raised when an operation lacks necessary permissions."""
    pass

class ValidationError(SwarmBaseException):
    """Raised when input validation fails."""
    pass

class APIError(SwarmBaseException):
    """Raised for errors related to API interactions."""
    pass

class ModelError(SwarmBaseException):
    """Raised for errors in model processing."""
    pass

def handle_exception(exception: Exception, logger=None, default_message="An unexpected error occurred"):
    """
    Centralized exception handler with optional logging.
    
    Args:
        exception (Exception): The caught exception
        logger (logging.Logger, optional): Logger to use
        default_message (str, optional): Default error message
    
    Returns:
        dict: Standardized error response
    """
    error_response = {
        "error": True,
        "type": type(exception).__name__,
        "message": str(exception) or default_message,
    }
    
    if logger:
        logger.error(f"{error_response['type']}: {error_response['message']}")
    
    return error_response