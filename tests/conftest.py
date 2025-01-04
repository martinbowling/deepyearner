"""
Pytest configuration and shared fixtures.
"""
import os
import tempfile
import pytest
from typing import Generator, Any
import sqlite3
from unittest.mock import MagicMock
from config import Config, DatabaseConfig, TwitterConfig, AnalysisConfig, LoggingConfig

@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file"""
    fd, path = tempfile.mkstemp(suffix='.db')
    yield path
    os.close(fd)
    os.unlink(path)

@pytest.fixture
def test_config(temp_db_path: str) -> Config:
    """Create a test configuration"""
    config = Config()
    
    # Override database config
    config.database = DatabaseConfig(
        path=temp_db_path,
        max_connections=2,
        timeout=5.0,
        retry_attempts=2,
        retry_delay=0.1
    )
    
    # Override Twitter config
    config.twitter = TwitterConfig(
        tweet_batch_size=10,
        max_daily_tweets=5,
        max_daily_follows=5,
        max_daily_likes=5,
        rate_limit_buffer=0.1
    )
    
    # Override analysis config
    config.analysis = AnalysisConfig(
        min_tweets_analyze=5,
        max_tweets_analyze=10,
        engagement_weight=0.4,
        topic_weight=0.4,
        interaction_weight=0.2,
        auto_follow_threshold=0.8,
        high_engagement_threshold=0.7
    )
    
    # Override logging config
    config.logging = LoggingConfig(
        log_level="DEBUG",
        log_file=os.path.join(tempfile.gettempdir(), "test.log"),
        max_size=1024,
        backup_count=1,
        json_format=True
    )
    
    return config

@pytest.fixture
def mock_twitter_client() -> MagicMock:
    """Create a mock Twitter client"""
    mock = MagicMock()
    
    # Set up common mock responses
    mock.get_user_by_id.return_value = {
        "data": {
            "id": "123",
            "username": "test_user",
            "name": "Test User"
        }
    }
    
    mock.get_users_tweets.return_value = {
        "data": [
            {
                "id": "1",
                "text": "Test tweet 1",
                "public_metrics": {
                    "retweet_count": 5,
                    "reply_count": 2,
                    "like_count": 10,
                    "quote_count": 1
                }
            }
        ]
    }
    
    return mock

@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client"""
    mock = MagicMock()
    
    # Set up common mock responses
    mock.analyze_content.return_value = {
        "sentiment": "positive",
        "topics": ["technology", "ai"],
        "engagement_score": 0.8
    }
    
    return mock

@pytest.fixture
def test_db(temp_db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Create a test database connection"""
    conn = sqlite3.connect(temp_db_path)
    conn.row_factory = sqlite3.Row
    
    # Create test tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            value INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS oauth_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            token_type TEXT,
            expires_at INTEGER
        );
    """)
    
    yield conn
    conn.close() 