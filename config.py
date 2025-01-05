"""
Central configuration management for DeepYearner.
Handles loading and validating all configuration values.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    path: str = "bot.db"
    max_connections: int = 5
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class TwitterConfig:
    """Twitter API configuration settings"""
    tweet_batch_size: int = 100
    max_daily_tweets: int = 50
    max_daily_follows: int = 50
    max_daily_likes: int = 100
    rate_limit_buffer: float = 0.1  # 10% buffer on rate limits

@dataclass
class AnalysisConfig:
    """User analysis configuration settings"""
    min_tweets_analyze: int = 20
    max_tweets_analyze: int = 100
    engagement_weight: float = 0.4
    topic_weight: float = 0.4
    interaction_weight: float = 0.2
    auto_follow_threshold: float = 0.8
    high_engagement_threshold: float = 0.7

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    log_level: str = "INFO"
    log_file: str = "bot.log"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    json_format: bool = True

@dataclass
class VectorDBConfig:
    """Vector database configuration settings"""
    path: str = "vector_db"
    dimension: int = 768  # Default for many embedding models
    max_elements: int = 10000
    ef_construction: int = 200
    M: int = 16

class Config:
    """Central configuration management"""
    
    def __init__(self):
        """Initialize configuration"""
        load_dotenv()
        
        self.database = DatabaseConfig()
        self.twitter = TwitterConfig()
        self.analysis = AnalysisConfig()
        self.logging = LoggingConfig()
        self.vector_db = VectorDBConfig()
        
        # Load environment overrides
        self._load_env_overrides()
        
        # Validate configuration
        self._validate_config()
    
    def _load_env_overrides(self):
        """Load overrides from environment variables"""
        # Database overrides
        if db_path := os.getenv('DB_PATH'):
            self.database.path = db_path
        if max_conn := os.getenv('DB_MAX_CONNECTIONS'):
            self.database.max_connections = int(max_conn)
            
        # Vector DB overrides
        if vector_db_path := os.getenv('VECTOR_DB_PATH'):
            self.vector_db.path = vector_db_path
        if vector_dim := os.getenv('VECTOR_DB_DIMENSION'):
            self.vector_db.dimension = int(vector_dim)
            
        # Twitter overrides
        if batch_size := os.getenv('TWITTER_BATCH_SIZE'):
            self.twitter.tweet_batch_size = int(batch_size)
        if daily_tweets := os.getenv('MAX_DAILY_TWEETS'):
            self.twitter.max_daily_tweets = int(daily_tweets)
            
        # Analysis overrides
        if auto_follow := os.getenv('AUTO_FOLLOW_THRESHOLD'):
            self.analysis.auto_follow_threshold = float(auto_follow)
        if eng_threshold := os.getenv('HIGH_ENGAGEMENT_THRESHOLD'):
            self.analysis.high_engagement_threshold = float(eng_threshold)
            
        # Logging overrides
        if log_level := os.getenv('LOG_LEVEL'):
            self.logging.log_level = log_level
        if log_file := os.getenv('LOG_FILE'):
            self.logging.log_file = log_file
    
    def _validate_config(self):
        """Validate configuration values"""
        # Database validation
        assert self.database.max_connections > 0, "Max connections must be positive"
        assert self.database.timeout > 0, "Database timeout must be positive"
        
        # Vector DB validation
        assert self.vector_db.dimension > 0, "Vector dimension must be positive"
        assert self.vector_db.max_elements > 0, "Max elements must be positive"
        assert self.vector_db.ef_construction > 0, "ef_construction must be positive"
        assert self.vector_db.M > 0, "M must be positive"
        
        # Twitter validation
        assert 0 < self.twitter.tweet_batch_size <= 100, "Tweet batch size must be between 1 and 100"
        assert self.twitter.max_daily_tweets > 0, "Max daily tweets must be positive"
        
        # Analysis validation
        assert 0 <= self.analysis.engagement_weight <= 1, "Engagement weight must be between 0 and 1"
        assert 0 <= self.analysis.topic_weight <= 1, "Topic weight must be between 0 and 1"
        assert 0 <= self.analysis.interaction_weight <= 1, "Interaction weight must be between 0 and 1"
        total_weight = (self.analysis.engagement_weight + 
                       self.analysis.topic_weight + 
                       self.analysis.interaction_weight)
        assert abs(total_weight - 1.0) < 0.001, "Analysis weights must sum to 1.0"
        
        # Logging validation
        assert self.logging.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], \
            "Invalid log level"
        assert self.logging.max_size > 0, "Log max size must be positive"
        assert self.logging.backup_count >= 0, "Backup count must be non-negative"

# Global configuration instance
config = Config()

# Export commonly used paths
VECTOR_DB_PATH = config.vector_db.path

# Load environment variables
load_dotenv()

# Twitter API Configuration
TWITTER_CLIENT_ID = os.getenv('TWITTER_CLIENT_ID')
TWITTER_CLIENT_SECRET = os.getenv('TWITTER_CLIENT_SECRET')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Vector Store Configuration
VECTOR_DB_PATH = Path("vector_store")
