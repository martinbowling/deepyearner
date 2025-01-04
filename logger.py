"""
Centralized logging configuration for DeepYearner.
Provides structured JSON logging with rotation and consistent formatting.
"""
import logging
import logging.handlers
import json
from datetime import datetime
from typing import Any, Dict
from config import config

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string"""
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": str(record.exc_info[0].__name__),
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
            
        return json.dumps(log_data)

def setup_logging() -> None:
    """Set up the logging configuration"""
    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging.log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = []
    
    # File handler with rotation
    if config.logging.json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler = logging.handlers.RotatingFileHandler(
        config.logging.log_file,
        maxBytes=config.logging.max_size,
        backupCount=config.logging.backup_count
    )
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    
    # Console handler (always use standard formatting for readability)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # Add all handlers
    for handler in handlers:
        root_logger.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)

class LoggerAdapter(logging.LoggerAdapter):
    """Custom adapter for adding context to log messages"""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Process the logging message and keyword arguments"""
        extra = kwargs.get("extra", {})
        if not "extra" in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(self.extra)
        return msg, kwargs

def get_logger_with_context(name: str, **context: Any) -> LoggerAdapter:
    """Get a logger with additional context fields"""
    logger = get_logger(name)
    return LoggerAdapter(logger, context)

# Set up logging when module is imported
setup_logging() 