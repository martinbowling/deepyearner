"""
Integration tests for critical system paths.
"""
import pytest
from unittest.mock import MagicMock
from typing import Generator
import os
import tempfile
from config import Config
from database import DatabaseManager
from twitter_utils import TwitterClient
from user_analyzer import UserAnalyzer
from mock_responses import get_mock_llm_response
from logger import get_logger

logger = get_logger(__name__)

@pytest.fixture
def setup_test_environment(test_config: Config) -> Generator[tuple[DatabaseManager, TwitterClient, UserAnalyzer], None, None]:
    """Set up test environment with all components"""
    # Initialize database
    db = DatabaseManager()
    
    # Create mock Twitter client
    twitter_client = MagicMock()
    twitter_client.get_user_by_id.return_value = {
        "data": {
            "id": "123",
            "username": "test_user",
            "name": "Test User"
        }
    }
    twitter_client.get_users_tweets.return_value = {
        "data": [
            {
                "id": "1",
                "text": "Test tweet about AI and technology",
                "public_metrics": {
                    "retweet_count": 10,
                    "reply_count": 5,
                    "like_count": 20,
                    "quote_count": 2
                }
            }
        ]
    }
    
    # Create mock LLM client
    llm_client = MagicMock()
    llm_client.analyze_content.return_value = get_mock_llm_response("content", "positive_tweet")
    
    # Initialize user analyzer
    analyzer = UserAnalyzer(twitter_client, llm_client)
    
    yield db, twitter_client, analyzer
    
    # Cleanup
    db.close()

def test_user_analysis_flow(setup_test_environment: tuple[DatabaseManager, TwitterClient, UserAnalyzer]):
    """Test complete user analysis flow"""
    db, twitter_client, analyzer = setup_test_environment
    
    # Add user to analysis queue
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_queue (username, priority, notes)
            VALUES (?, ?, ?)
        """, ("test_user", 1, "Test analysis"))
    
    # Analyze user
    result = analyzer.analyze_user("test_user")
    
    # Verify analysis results
    assert result.is_success
    assert result.data is not None
    assert result.data.get("engagement_score", 0) > 0
    
    # Verify database was updated
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM user_analysis
            WHERE username = ?
        """, ("test_user",))
        analysis_record = cursor.fetchone()
        
        assert analysis_record is not None
        assert analysis_record["engagement_score"] > 0

def test_error_handling_flow(setup_test_environment: tuple[DatabaseManager, TwitterClient, UserAnalyzer]):
    """Test error handling in critical paths"""
    db, twitter_client, analyzer = setup_test_environment
    
    # Simulate API error
    twitter_client.get_users_tweets.side_effect = Exception("API Error")
    
    # Attempt analysis
    result = analyzer.analyze_user("test_user")
    
    # Verify error handling
    assert not result.is_success
    assert len(result.errors) > 0
    assert "API Error" in str(result.errors[0])
    
    # Verify error was logged
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT status FROM analysis_queue
            WHERE username = ?
        """, ("test_user",))
        status = cursor.fetchone()
        
        assert status is not None
        assert status["status"] == "error"

def test_rate_limit_handling(setup_test_environment: tuple[DatabaseManager, TwitterClient, UserAnalyzer]):
    """Test rate limit handling"""
    db, twitter_client, analyzer = setup_test_environment
    
    # Set up rate limit sequence
    twitter_client.get_users_tweets.side_effect = [
        Exception("Rate limit exceeded"),  # First call fails
        {  # Second call succeeds
            "data": [{
                "id": "1",
                "text": "Test tweet",
                "public_metrics": {
                    "retweet_count": 5,
                    "reply_count": 2,
                    "like_count": 10,
                    "quote_count": 1
                }
            }]
        }
    ]
    
    # Analyze user
    result = analyzer.analyze_user("test_user")
    
    # Verify successful retry
    assert result.is_success
    assert len(result.warnings) > 0
    assert "Rate limit" in str(result.warnings[0])

def test_partial_results_handling(setup_test_environment: tuple[DatabaseManager, TwitterClient, UserAnalyzer]):
    """Test handling of partial results"""
    db, twitter_client, analyzer = setup_test_environment
    
    # Return partial data
    twitter_client.get_users_tweets.return_value = {
        "data": []  # No tweets
    }
    
    # Analyze user
    result = analyzer.analyze_user("test_user")
    
    # Verify partial result handling
    assert result.is_partial
    assert result.data is not None
    assert len(result.warnings) > 0
    assert "No tweets found" in str(result.warnings[0])

def test_concurrent_analysis(setup_test_environment: tuple[DatabaseManager, TwitterClient, UserAnalyzer]):
    """Test concurrent user analysis"""
    db, twitter_client, analyzer = setup_test_environment
    
    # Add multiple users to queue
    users = ["user1", "user2", "user3", "user4", "user5"]
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for user in users:
            cursor.execute("""
                INSERT INTO analysis_queue (username, priority)
                VALUES (?, ?)
            """, (user, 1))
    
    # Analyze users concurrently
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(analyzer.analyze_user, user) for user in users]
        results = [future.result() for future in futures]
    
    # Verify all analyses completed
    assert len(results) == len(users)
    assert all(result.is_success or result.is_partial for result in results)
    
    # Verify database updates
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM user_analysis")
        count = dict(cursor.fetchone())["count"]
        assert count == len(users) 