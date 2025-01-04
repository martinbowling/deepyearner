"""
Central configuration management for DeepYearner.
Handles loading and validating all configuration values.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

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

class Config:
    """Central configuration management"""
    
    def __init__(self):
        """Initialize configuration"""
        load_dotenv()
        
        self.database = DatabaseConfig()
        self.twitter = TwitterConfig()
        self.analysis = AnalysisConfig()
        self.logging = LoggingConfig()
        
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
