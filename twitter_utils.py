"""
Twitter client abstraction that handles all Twitter API interactions.
Provides a unified interface for both read and write operations.
"""
import os
import time
import logging
from typing import Dict, List, Optional, Union
import tweepy
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RateLimiter:
    """Handles rate limiting for Twitter API calls"""
    def __init__(self):
        self.limits = {
            'tweets': {'window': 3600, 'max': 300, 'used': 0, 'reset': time.time()},
            'replies': {'window': 3600, 'max': 250, 'used': 0, 'reset': time.time()},
            'likes': {'window': 3600, 'max': 500, 'used': 0, 'reset': time.time()},
            'follows': {'window': 3600, 'max': 400, 'used': 0, 'reset': time.time()},
            'dms': {'window': 3600, 'max': 200, 'used': 0, 'reset': time.time()}
        }
    
    def check_and_update(self, action_type: str) -> bool:
        """Check if action is allowed and update counters"""
        now = time.time()
        limit = self.limits[action_type]
        
        # Reset if window expired
        if now > limit['reset']:
            limit['used'] = 0
            limit['reset'] = now + limit['window']
        
        # Check if action allowed
        if limit['used'] >= limit['max']:
            return False
        
        limit['used'] += 1
        return True

class TwitterClient:
    """Unified Twitter client for all API interactions"""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_secret: str):
        """Initialize Twitter client with credentials"""
        self.auth = tweepy.OAuthHandler(api_key, api_secret)
        self.auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        self.client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret,
            wait_on_rate_limit=True
        )
        self.rate_limiter = RateLimiter()
        self.user_id = self.client.get_me().data.id
    
    # Tweet Operations
    def create_tweet(self, text: str, reply_to_id: Optional[str] = None, 
                    quote_tweet_id: Optional[str] = None) -> Optional[str]:
        """Create a new tweet, reply, or quote tweet"""
        try:
            if not self.rate_limiter.check_and_update('tweets'):
                logger.warning("Tweet rate limit reached")
                return None
            
            response = self.client.create_tweet(
                text=text,
                in_reply_to_tweet_id=reply_to_id,
                quote_tweet_id=quote_tweet_id
            )
            return str(response.data['id'])
        except Exception as e:
            logger.error(f"Error creating tweet: {str(e)}")
            return None
    
    def post_thread(self, tweets: List[str]) -> List[str]:
        """Post a thread of tweets"""
        thread_ids = []
        reply_to = None
        
        for tweet in tweets:
            tweet_id = self.create_tweet(tweet, reply_to_id=reply_to)
            if tweet_id:
                thread_ids.append(tweet_id)
                reply_to = tweet_id
            else:
                break
        
        return thread_ids
    
    def retweet(self, tweet_id: str) -> bool:
        """Retweet a tweet"""
        try:
            if not self.rate_limiter.check_and_update('tweets'):
                return False
            
            self.client.retweet(tweet_id)
            return True
        except Exception as e:
            logger.error(f"Error retweeting: {str(e)}")
            return False
    
    def like_tweet(self, tweet_id: str) -> bool:
        """Like a tweet"""
        try:
            if not self.rate_limiter.check_and_update('likes'):
                return False
            
            self.client.like(tweet_id)
            return True
        except Exception as e:
            logger.error(f"Error liking tweet: {str(e)}")
            return False
    
    # Timeline & Search Operations
    def get_home_timeline(self, max_results: int = 100) -> List[Dict]:
        """Get tweets from home timeline"""
        try:
            tweets = []
            for tweet in tweepy.Paginator(
                self.client.get_home_timeline,
                max_results=max_results
            ).flatten(limit=max_results):
                tweets.append(tweet.data)
            return tweets
        except Exception as e:
            logger.error(f"Error getting timeline: {str(e)}")
            return []
    
    def search_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search for tweets"""
        try:
            tweets = []
            for tweet in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=max_results
            ).flatten(limit=max_results):
                tweets.append(tweet.data)
            return tweets
        except Exception as e:
            logger.error(f"Error searching tweets: {str(e)}")
            return []
    
    def get_user_tweets(self, user_id: str, max_results: int = 100) -> List[Dict]:
        """Get tweets from a specific user"""
        try:
            tweets = []
            for tweet in tweepy.Paginator(
                self.client.get_users_tweets,
                id=user_id,
                max_results=max_results
            ).flatten(limit=max_results):
                tweets.append(tweet.data)
            return tweets
        except Exception as e:
            logger.error(f"Error getting user tweets: {str(e)}")
            return []
    
    # User Operations
    def follow_user(self, user_id: str) -> bool:
        """Follow a user"""
        try:
            if not self.rate_limiter.check_and_update('follows'):
                return False
            
            self.client.follow_user(user_id)
            return True
        except Exception as e:
            logger.error(f"Error following user: {str(e)}")
            return False
    
    def unfollow_user(self, user_id: str) -> bool:
        """Unfollow a user"""
        try:
            if not self.rate_limiter.check_and_update('follows'):
                return False
            
            self.client.unfollow_user(user_id)
            return True
        except Exception as e:
            logger.error(f"Error unfollowing user: {str(e)}")
            return False
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user information by username"""
        try:
            response = self.client.get_user(username=username)
            return response.data
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return None
    
    def get_user_followers(self, user_id: str, max_results: int = 100) -> List[Dict]:
        """Get user's followers"""
        try:
            followers = []
            for follower in tweepy.Paginator(
                self.client.get_users_followers,
                id=user_id,
                max_results=max_results
            ).flatten(limit=max_results):
                followers.append(follower.data)
            return followers
        except Exception as e:
            logger.error(f"Error getting followers: {str(e)}")
            return []
    
    def get_user_following(self, user_id: str, max_results: int = 100) -> List[Dict]:
        """Get users that a user is following"""
        try:
            following = []
            for user in tweepy.Paginator(
                self.client.get_users_following,
                id=user_id,
                max_results=max_results
            ).flatten(limit=max_results):
                following.append(user.data)
            return following
        except Exception as e:
            logger.error(f"Error getting following: {str(e)}")
            return []
    
    # Direct Message Operations
    def send_dm(self, recipient_id: str, text: str) -> bool:
        """Send a direct message"""
        try:
            if not self.rate_limiter.check_and_update('dms'):
                return False
            
            self.client.create_direct_message(participant_id=recipient_id, text=text)
            return True
        except Exception as e:
            logger.error(f"Error sending DM: {str(e)}")
            return False
    
    def get_dms(self, max_results: int = 50) -> List[Dict]:
        """Get direct messages"""
        try:
            return self.client.get_direct_messages(max_results=max_results)
        except Exception as e:
            logger.error(f"Error getting DMs: {str(e)}")
            return []

def get_twitter_client() -> TwitterClient:
    """Factory function to create TwitterClient instance"""
    return TwitterClient(
        api_key=os.getenv('TWITTER_API_KEY'),
        api_secret=os.getenv('TWITTER_API_SECRET'),
        access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
        access_secret=os.getenv('TWITTER_ACCESS_SECRET')
    )
