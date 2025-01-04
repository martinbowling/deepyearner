"""
Twitter client abstraction that handles all Twitter API interactions.
Provides a unified interface for both read and write operations.
"""
import os
import time
import logging
import requests
from typing import Dict, List, Optional, Union, Set
import tweepy
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def get_oauth2_tokens():
    """Get OAuth 2.0 tokens from the database"""
    try:
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('SELECT access_token, refresh_token, expires_at FROM oauth_tokens WHERE id = 1')
        row = c.fetchone()
        conn.close()
        
        if not row:
            return None, None, None
            
        return row[0], row[1], row[2]
    except Exception as e:
        logger.error(f"Error getting OAuth tokens: {str(e)}")
        return None, None, None

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
    """Twitter API client wrapper"""
    
    def __init__(self):
        """Initialize Twitter API client with credentials."""
        load_dotenv()
        
        # Get OAuth 2.0 tokens from database
        access_token, refresh_token, expires_at = get_oauth2_tokens()
        
        if access_token:
            # Use OAuth 2.0 token
            self.auth_headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'User-Agent': 'DeepYearnerBot/1.0'
            }
            self.api_base = 'https://api.twitter.com/2'
            self.oauth2_mode = True
            logger.info("Using OAuth 2.0 authentication")
        else:
            # Fall back to OAuth 1.0a credentials
            logger.info("Falling back to OAuth 1.0a authentication")
            self.client = tweepy.Client(
                bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
                consumer_key=os.getenv('TWITTER_API_KEY'),
                consumer_secret=os.getenv('TWITTER_API_SECRET'),
                access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
                access_token_secret=os.getenv('TWITTER_ACCESS_SECRET'),
                wait_on_rate_limit=True
            )
            self.oauth2_mode = False
        
        self.user_id = None
        self.rate_limiter = RateLimiter()
        self._initialize_user()

    def _initialize_user(self) -> None:
        """Get and store the authenticated user's ID."""
        try:
            me = self.get_me()
            if me and 'data' in me:
                self.user_id = me['data']['id']
        except Exception as e:
            logger.error(f"Error initializing user: {str(e)}")
            raise

    def get_me(self):
        """Get authenticated user info"""
        try:
            if self.oauth2_mode:
                response = requests.get(
                    f"{self.api_base}/users/me",
                    headers=self.auth_headers
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Error getting user info: {response.text}")
                    return None
            else:
                return self.client.get_me()
        except Exception as e:
            logger.error(f"Error getting user info: {str(e)}")
            return None
        
    def get_user_by_username(self, username: str):
        """Get user by username"""
        try:
            if self.oauth2_mode:
                response = requests.get(
                    f"{self.api_base}/users/by/username/{username}",
                    headers=self.auth_headers,
                    params={
                        'user.fields': 'description,public_metrics,created_at,profile_image_url'
                    }
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Error getting user by username: {response.text}")
                    return None
            else:
                return self.client.get_user(
                    username=username,
                    user_fields=['description', 'public_metrics', 'created_at', 'profile_image_url']
                )
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
        
    def get_user_by_id(self, user_id: Union[str, int]):
        """Get user by ID"""
        try:
            return self.client.get_user(
                id=user_id,
                user_fields=['description', 'public_metrics', 'created_at', 'profile_image_url']
            )
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
        
    def get_users_tweets(self, user_id: Optional[Union[str, int]] = None, max_results: int = 10):
        """Get tweets for a user"""
        try:
            user_id = user_id or self.user_id
            if self.oauth2_mode:
                response = requests.get(
                    f"{self.api_base}/users/{user_id}/tweets",
                    headers=self.auth_headers,
                    params={
                        'max_results': max_results,
                        'tweet.fields': 'created_at,public_metrics,context_annotations',
                        'expansions': 'attachments.media_keys,referenced_tweets.id'
                    }
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Error getting user tweets: {response.text}")
                    return None
            else:
                return self.client.get_users_tweets(
                    id=user_id,
                    max_results=max_results,
                    tweet_fields=['created_at', 'public_metrics', 'context_annotations'],
                    expansions=['attachments.media_keys', 'referenced_tweets.id'],
                )
        except Exception as e:
            logger.error(f"Error getting user tweets: {str(e)}")
            return None
        
    def get_tweet(self, tweet_id: Union[str, int]):
        """Get a single tweet"""
        try:
            return self.client.get_tweet(
                id=tweet_id,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            )
        except Exception as e:
            logger.error(f"Error getting tweet: {str(e)}")
            return None
        
    def create_tweet(self, text: str, reply_to: Optional[Union[str, int]] = None):
        """Create a new tweet"""
        if not self.rate_limiter.check_and_update('tweets'):
            logger.warning("Rate limit exceeded for tweets")
            return None
            
        try:
            if self.oauth2_mode:
                data = {'text': text}
                if reply_to:
                    data['reply'] = {'in_reply_to_tweet_id': str(reply_to)}
                
                response = requests.post(
                    f"{self.api_base}/tweets",
                    headers=self.auth_headers,
                    json=data
                )
                if response.status_code == 201:  # Twitter API returns 201 for successful creation
                    return response.json()
                else:
                    logger.error(f"Error creating tweet: {response.text}")
                    return None
            else:
                return self.client.create_tweet(
                    text=text,
                    in_reply_to_tweet_id=reply_to
                )
        except Exception as e:
            logger.error(f"Error creating tweet: {str(e)}")
            return None
        
    def like_tweet(self, tweet_id: Union[str, int]):
        """Like a tweet"""
        if not self.rate_limiter.check_and_update('likes'):
            logger.warning("Rate limit exceeded for likes")
            return False
            
        try:
            self.client.like(tweet_id)
            return True
        except Exception as e:
            logger.error(f"Error liking tweet: {str(e)}")
            return False
        
    def unlike_tweet(self, tweet_id: Union[str, int]):
        """Unlike a tweet"""
        try:
            self.client.unlike(tweet_id)
            return True
        except Exception as e:
            logger.error(f"Error unliking tweet: {str(e)}")
            return False
        
    def retweet(self, tweet_id: Union[str, int]):
        """Retweet a tweet"""
        try:
            self.client.retweet(tweet_id)
            return True
        except Exception as e:
            logger.error(f"Error retweeting: {str(e)}")
            return False
        
    def unretweet(self, tweet_id: Union[str, int]):
        """Undo a retweet"""
        try:
            self.client.unretweet(tweet_id)
            return True
        except Exception as e:
            logger.error(f"Error unretweeting: {str(e)}")
            return False
        
    def follow_user(self, user_id: Union[str, int]):
        """Follow a user"""
        if not self.rate_limiter.check_and_update('follows'):
            logger.warning("Rate limit exceeded for follows")
            return False
            
        try:
            self.client.follow_user(user_id)
            return True
        except Exception as e:
            logger.error(f"Error following user: {str(e)}")
            return False
        
    def unfollow_user(self, user_id: Union[str, int]):
        """Unfollow a user"""
        try:
            self.client.unfollow_user(user_id)
            return True
        except Exception as e:
            logger.error(f"Error unfollowing user: {str(e)}")
            return False
        
    def search_tweets(self, query: str, max_results: int = 10):
        """Search for tweets"""
        try:
            return self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'lang', 'author_id'],
                user_fields=['username', 'profile_image_url'],
                expansions=['author_id']
            )
        except Exception as e:
            logger.error(f"Error searching tweets: {str(e)}")
            return None
        
    def get_user_timeline(self, user_id: Optional[Union[str, int]] = None, max_results: int = 10):
        """Get user timeline"""
        return self.get_users_tweets(user_id, max_results)
        
    def get_user_mentions(self, user_id: Optional[Union[str, int]] = None, max_results: int = 10):
        """Get mentions of a user"""
        try:
            user_id = user_id or self.user_id
            return self.client.get_users_mentions(
                id=user_id,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            )
        except Exception as e:
            logger.error(f"Error getting mentions: {str(e)}")
            return None
        
    def get_tweet_liking_users(self, tweet_id: Union[str, int], max_results: int = 100):
        """Get users who liked a tweet"""
        try:
            return self.client.get_liking_users(
                id=tweet_id,
                max_results=max_results,
                user_fields=['description', 'public_metrics', 'profile_image_url']
            )
        except Exception as e:
            logger.error(f"Error getting liking users: {str(e)}")
            return None
        
    def get_tweet_retweeting_users(self, tweet_id: Union[str, int], max_results: int = 100):
        """Get users who retweeted a tweet"""
        try:
            return self.client.get_retweeters(
                id=tweet_id,
                max_results=max_results,
                user_fields=['description', 'public_metrics', 'profile_image_url']
            )
        except Exception as e:
            logger.error(f"Error getting retweeting users: {str(e)}")
            return None

    def _validate_token(self):
        """Validate and refresh OAuth 2.0 token if needed"""
        try:
            if not self.oauth2_mode:
                return True
            
            # Get current token info
            access_token, refresh_token, expires_at = get_oauth2_tokens()
            
            # Check if token is expired or will expire soon
            if expires_at and datetime.fromisoformat(expires_at) < datetime.now() + timedelta(minutes=5):
                # Refresh token
                response = requests.post(
                    'https://api.twitter.com/2/oauth2/token',
                    headers={
                        'Content-Type': 'application/x-www-form-urlencoded'
                    },
                    data={
                        'refresh_token': refresh_token,
                        'grant_type': 'refresh_token',
                        'client_id': os.getenv('OAUTH_CLIENT_ID')
                    }
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    # Update tokens in database
                    conn = sqlite3.connect('bot.db')
                    c = conn.cursor()
                    c.execute('''
                        UPDATE oauth_tokens 
                        SET access_token = ?, refresh_token = ?, expires_at = ?
                        WHERE id = 1
                    ''', (
                        token_data['access_token'],
                        token_data['refresh_token'],
                        (datetime.now() + timedelta(seconds=token_data['expires_in'])).isoformat()
                    ))
                    conn.commit()
                    conn.close()
                    
                    # Update current client
                    self.auth_headers['Authorization'] = f"Bearer {token_data['access_token']}"
                    return True
                else:
                    logger.error(f"Error refreshing token: {response.text}")
                    return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return False

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make an API request with token validation"""
        try:
            if not self._validate_token():
                return None
            
            if 'headers' not in kwargs:
                kwargs['headers'] = {}
            
            # Ensure we're using OAuth 2.0 User Context
            access_token = self.auth_headers['Authorization'].split(' ')[1]
            kwargs['headers'].update({
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'User-Agent': 'DeepYearnerBot/1.0'
            })
            
            response = requests.request(
                method,
                f"{self.api_base}{endpoint}",
                **kwargs
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Token might be expired, try to refresh
                if self._validate_token():
                    # Retry with new token
                    return self._make_request(method, endpoint, **kwargs)
                return None
            else:
                logger.error(f"Error making request to {endpoint}: {response.text}")
                return None
            
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {str(e)}")
            return None

def get_twitter_client() -> TwitterClient:
    """Factory function to create TwitterClient instance"""
    return TwitterClient()
