"""
Retry decorator with configurable backoff for handling transient errors.
"""
import time
import random
from functools import wraps
from typing import Type, Tuple, Optional, Callable, Any
from logger import get_logger
from exceptions import DeepYearnerError, LLMRateLimitError, TwitterAPIError

logger = get_logger(__name__)

def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    should_retry: Optional[Callable[[Exception], bool]] = None
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: Whether to use exponential backoff
        jitter: Whether to add random jitter to delay
        exceptions: Tuple of exceptions to catch and retry
        should_retry: Optional function to determine if retry should occur
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if should_retry and not should_retry(e):
                        raise
                    
                    # Don't retry on last attempt
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Calculate delay
                    if exponential:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    # Add jitter if requested
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    # Handle rate limit errors specially
                    if isinstance(e, (LLMRateLimitError, TwitterAPIError)):
                        if getattr(e, 'retry_after', None):
                            delay = e.retry_after
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed, retrying in {delay:.2f}s",
                        extra={
                            "error": str(last_exception),
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "delay": delay
                        }
                    )
                    
                    time.sleep(delay)
            
            # We should never get here, but just in case
            raise last_exception or DeepYearnerError("Retry failed")
        
        return wrapper
    
    return decorator

def should_retry_api_call(error: Exception) -> bool:
    """Determine if an API error should be retried"""
    if isinstance(error, TwitterAPIError):
        # Don't retry auth errors or not found errors
        if error.details.get("error_type") in ["auth", "not_found"]:
            return False
    return True 