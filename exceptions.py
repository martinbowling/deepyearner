"""
Centralized exception handling for DeepYearner.
Defines custom exceptions and provides consistent error handling strategies.
"""
from typing import Optional, Dict, Any
from logger import get_logger

logger = get_logger(__name__)

class DeepYearnerError(Exception):
    """Base exception class for all DeepYearner errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
        # Log the error with context
        logger.error(
            self.message,
            extra={
                "error_type": self.__class__.__name__,
                "details": self.details
            }
        )

class LLMError(DeepYearnerError):
    """Base class for LLM-related errors"""
    pass

class LLMConnectionError(LLMError):
    """Error connecting to LLM service"""
    pass

class LLMTimeoutError(LLMError):
    """LLM request timed out"""
    pass

class LLMRateLimitError(LLMError):
    """Hit rate limit for LLM service"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after

class LLMResponseError(LLMError):
    """Error processing LLM response"""
    pass

class LLMValidationError(LLMError):
    """LLM response failed validation"""
    pass

class APIError(DeepYearnerError):
    """Base class for API-related errors"""
    pass

class TwitterAPIError(APIError):
    """Error interacting with Twitter API"""
    pass

class DatabaseError(DeepYearnerError):
    """Base class for database errors"""
    pass

class ConfigError(DeepYearnerError):
    """Configuration-related errors"""
    pass

class ValidationError(DeepYearnerError):
    """Data validation errors"""
    pass

def handle_llm_error(error: Exception) -> LLMError:
    """Convert various LLM provider errors to our standard exceptions"""
    # Example error mapping for different LLM providers
    if isinstance(error, TimeoutError):
        return LLMTimeoutError("LLM request timed out", details={"original_error": str(error)})
    elif "rate limit" in str(error).lower():
        return LLMRateLimitError("Rate limit exceeded", retry_after=None)
    elif "connection" in str(error).lower():
        return LLMConnectionError(f"Failed to connect to LLM service: {str(error)}")
    else:
        return LLMError(f"Unexpected LLM error: {str(error)}")

def handle_twitter_error(error: Exception) -> TwitterAPIError:
    """Convert Twitter API errors to our standard exceptions"""
    error_msg = str(error)
    details = {"original_error": error_msg}
    
    if "rate limit" in error_msg.lower():
        details["error_type"] = "rate_limit"
    elif "unauthorized" in error_msg.lower():
        details["error_type"] = "auth"
    elif "not found" in error_msg.lower():
        details["error_type"] = "not_found"
    else:
        details["error_type"] = "unknown"
    
    return TwitterAPIError(f"Twitter API error: {error_msg}", details=details) 