"""
Results handler for managing partial results and fallback behaviors.
"""
from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class ResultStatus(Enum):
    """Status of a result"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"

@dataclass
class Result(Generic[T]):
    """Container for operation results with status and error information"""
    status: ResultStatus
    data: Optional[T]
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    
    @property
    def is_success(self) -> bool:
        return self.status == ResultStatus.SUCCESS
    
    @property
    def is_partial(self) -> bool:
        return self.status == ResultStatus.PARTIAL
    
    @property
    def is_failure(self) -> bool:
        return self.status == ResultStatus.FAILURE

class ResultHandler(Generic[T]):
    """Handler for managing operation results and fallbacks"""
    
    def __init__(self, fallback_strategy: Optional[Callable[[], T]] = None):
        self.fallback_strategy = fallback_strategy
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def add_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add an error to the result"""
        self.errors.append({
            "error": error,
            "details": details or {}
        })
        logger.error(error, extra={"details": details})
    
    def add_warning(self, warning: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add a warning to the result"""
        self.warnings.append({
            "warning": warning,
            "details": details or {}
        })
        logger.warning(warning, extra={"details": details})
    
    def handle_result(self, data: Optional[T] = None) -> Result[T]:
        """Create a result based on current state"""
        if not self.errors:
            return Result(
                status=ResultStatus.SUCCESS,
                data=data,
                errors=[],
                warnings=self.warnings
            )
        
        # Try fallback if available and complete failure
        if data is None and self.fallback_strategy:
            try:
                data = self.fallback_strategy()
                self.add_warning("Used fallback strategy", {"fallback_data": str(data)})
            except Exception as e:
                self.add_error("Fallback strategy failed", {"error": str(e)})
        
        # Determine final status
        if data is not None:
            status = ResultStatus.PARTIAL if self.errors else ResultStatus.SUCCESS
        else:
            status = ResultStatus.FAILURE
        
        return Result(
            status=status,
            data=data,
            errors=self.errors,
            warnings=self.warnings
        )

def with_fallback(fallback_value: T) -> Callable[[Callable[..., T]], Callable[..., Result[T]]]:
    """Decorator to add fallback behavior to a function"""
    def decorator(func: Callable[..., T]) -> Callable[..., Result[T]]:
        def wrapper(*args: Any, **kwargs: Any) -> Result[T]:
            handler = ResultHandler(lambda: fallback_value)
            try:
                result = func(*args, **kwargs)
                return handler.handle_result(result)
            except Exception as e:
                handler.add_error(str(e))
                return handler.handle_result()
        return wrapper
    return decorator

def combine_results(results: List[Result[T]]) -> Result[List[T]]:
    """Combine multiple results into a single result"""
    combined_handler = ResultHandler()
    combined_data: List[T] = []
    
    for result in results:
        if result.data is not None:
            combined_data.append(result.data)
        combined_handler.errors.extend(result.errors)
        combined_handler.warnings.extend(result.warnings)
    
    return combined_handler.handle_result(combined_data if combined_data else None) 