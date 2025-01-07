"""Custom exceptions for the Twitter bot"""

class RateLimitError(Exception):
    """Exception raised when rate limit is hit"""
    def __init__(self, message: str, reset_time: Optional[int] = None):
        self.message = message
        self.reset_time = reset_time
        super().__init__(self.message)

class AuthenticationError(Exception):
    """Exception raised when authentication fails"""
    pass

class APIError(Exception):
    """Exception raised for general API errors"""
    pass 