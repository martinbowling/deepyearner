"""
Twitter client abstraction that handles all Twitter API interactions.
Provides a unified interface for both read and write operations.
"""
import os
import time
import logging
import requests
from typing import Dict, List, Optional, Union, Set, Any
import tweepy
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
import contextlib
import asyncio
import aiohttp
import json
import hmac
import base64
import hashlib
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)

# Global connection lock
db_lock = threading.Lock()

@contextlib.contextmanager
def get_db_connection():
    """Get a database connection with timeout and proper locking"""
    with db_lock:
        conn = sqlite3.connect('bot.db', timeout=30)
        try:
            yield conn
        finally:
            conn.close()

def get_oauth2_tokens():
    """Get OAuth 2.0 tokens from the database"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT access_token, refresh_token, expires_at FROM oauth_tokens WHERE id = 1')
            row = c.fetchone()
            
        if not row:
            return None, None, None
            
        return row[0], row[1], row[2]
    except Exception as e:
        logger.error(f"Error getting OAuth tokens: {str(e)}")
        return None, None, None

class TokenRefresher(threading.Thread):
    """Background thread to refresh OAuth token periodically"""
    def __init__(self, client):
        super().__init__(daemon=True)
        self.client = client
        self.running = True
        self._last_refresh = 0
        
    def run(self):
        while self.running:
            try:
                current_time = time.time()
                # Add minimum interval between refresh attempts
                if current_time - self._last_refresh < 60:
                    time.sleep(5)
                    continue
                    
                # Check if token needs refresh (5 minutes before expiry)
                if (self.client.token_expiry and 
                    isinstance(self.client.token_expiry, datetime) and
                    self.client.token_expiry < datetime.now() + timedelta(minutes=5)):
                    logger.info("Background task refreshing token")
                    self.client._validate_token()
                    self._last_refresh = current_time
                
                # Sleep for 1 minute before next check
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in token refresh thread: {str(e)}")
                time.sleep(60)  # Sleep on error to avoid tight loop
    
    def stop(self):
        self.running = False

class TwitterClient:
    """Twitter API client wrapper"""
    
    def __init__(self):
        """Initialize Twitter API client"""
        self.api_base = "https://api.twitter.com/2"
        self.oauth2_mode = True
        self.auth_headers = None
        self.token_expiry = None
        self.recent_tweets = set()
        self.last_request_time = {}
        self.min_request_interval = {
            'timeline': 60,  # 1 minute between timeline requests
            'tweet': 30,     # 30 seconds between tweets
            'like': 10,      # 10 seconds between likes
            'retweet': 10    # 10 seconds between retweets
        }
        self._initialize_auth()
        
        # Start token refresh thread
        self.token_refresher = TokenRefresher(self)
        self.token_refresher.start()
        
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'token_refresher'):
            self.token_refresher.stop()
            self.token_refresher.join(timeout=1)

    def _initialize_auth(self):
        """Initialize authentication"""
        try:
            # Get OAuth tokens from database
            access_token, refresh_token, expires_at = get_oauth2_tokens()
            
            if access_token:
                logger.info("Using OAuth 2.0 authentication")
                self.oauth2_mode = True
                self.auth_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "DeepYearner/1.0"
                }
                # Set token expiry
                self.token_expiry = datetime.fromisoformat(expires_at) if expires_at else None
                
                # Initialize user info
                me = self.get_me()
                if me and 'data' in me:
                    self.user_id = me['data']['id']
                    logger.info(f"Authenticated as user {self.user_id}")
                else:
                    logger.error("Could not get user info")
                    raise Exception("Could not get user info - check OAuth scopes")
            else:
                logger.error("No authentication credentials found")
                raise Exception("No authentication credentials found - run oauth_setup.py first")
        except Exception as e:
            logger.error(f"Error initializing authentication: {str(e)}")
            raise

    def _validate_token(self):
        """Validate and refresh OAuth 2.0 token if needed"""
        try:
            if not self.oauth2_mode:
                return True
            
            # Check if token is expired or will expire soon
            if (self.token_expiry and 
                isinstance(self.token_expiry, datetime) and
                self.token_expiry < datetime.now() + timedelta(minutes=5)):
                logger.info("Token expiring soon, refreshing authentication")
                
                # Get current tokens
                access_token, refresh_token, expires_at = get_oauth2_tokens()
                
                if not refresh_token:
                    logger.error("No refresh token available")
                    return False
                
                # Refresh token
                auth = (os.getenv('OAUTH_CLIENT_ID'), os.getenv('OAUTH_CLIENT_SECRET'))
                response = requests.post(
                    'https://api.twitter.com/2/oauth2/token',
                    auth=auth,
                    headers={
                        'Content-Type': 'application/x-www-form-urlencoded'
                    },
                    data={
                        'refresh_token': refresh_token,
                        'grant_type': 'refresh_token'
                    }
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    # Update tokens in database with proper connection handling
                    with get_db_connection() as conn:
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
                    
                    # Update current client
                    self.auth_headers['Authorization'] = f"Bearer {token_data['access_token']}"
                    self.token_expiry = datetime.now() + timedelta(seconds=token_data['expires_in'])
                    return True
                else:
                    logger.error(f"Error refreshing token: {response.text}")
                    return False

            return True
            
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return False

    def get_me(self):
        """Get authenticated user info"""
        try:
            response = requests.get(
                f"{self.api_base}/users/me",
                headers=self.auth_headers,
                params={
                    'user.fields': 'description,public_metrics,created_at,profile_image_url'
                }
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("Unauthorized - check OAuth scopes and token validity")
                return None
            else:
                logger.error(f"Error getting user info: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting user info: {str(e)}")
            return None

    def create_tweet(self, text: str, reply_settings: Optional[str] = None, in_reply_to_tweet_id: Optional[str] = None):
        """Create a tweet"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Check for duplicate content
                if text in self.recent_tweets:
                    logger.warning("Duplicate tweet content detected")
                    return None
                    
                data = {
                    'text': text
                }
                
                if in_reply_to_tweet_id:
                    data['reply'] = {'in_reply_to_tweet_id': in_reply_to_tweet_id}
                    
                response = requests.post(
                    f"{self.api_base}/tweets",
                    headers=self.auth_headers,
                    json=data
                )
                if response.status_code == 201:
                    # Add to recent tweets cache
                    self.recent_tweets.add(text)
                    # Keep cache size reasonable
                    if len(self.recent_tweets) > 1000:
                        self.recent_tweets.pop()
                    return response.json()
                elif response.status_code == 429:
                    if self._handle_rate_limit(response):
                        retry_count += 1
                        continue
                else:
                    logger.error(f"Error creating tweet: {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error creating tweet: {str(e)}")
                return None
            retry_count += 1
        return None

    def _handle_rate_limit(self, response):
        """Handle rate limit response"""
        if response.status_code == 429:
            reset_time = int(response.headers.get('x-rate-limit-reset', 0))
            if reset_time:
                wait_time = max(reset_time - int(time.time()), 60)  # At least 60 seconds
                logger.warning(f"Rate limited. Waiting {wait_time} seconds until {reset_time}")
                time.sleep(wait_time)
                return True
        return False

    def get_users_tweets(self, user_id: Optional[str] = None, **params) -> Dict:
        """Get tweets from a user"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                if not user_id:
                    user_id = self.user_id
                response = requests.get(
                    f"{self.api_base}/users/{user_id}/tweets",
                    headers=self.auth_headers,
                    params=params
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    if self._handle_rate_limit(response):
                        retry_count += 1
                        continue
                else:
                    logger.error(f"Error getting user tweets: {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error getting user tweets: {str(e)}")
                return None
            retry_count += 1
        return None

    async def get_home_timeline(self):
        """Get home timeline with rate limiting"""
        try:
            await self._wait_for_rate_limit('timeline')
            # Make the API request with correct endpoint and parameters
            response = await self._make_request(
                'GET', 
                f'/2/users/{self.user_id}/timelines/reverse_chronological',  # Use user ID in endpoint
                params={
                    'tweet.fields': 'created_at,public_metrics,conversation_id,in_reply_to_user_id,author_id',
                    'max_results': 100,
                    'expansions': 'author_id,referenced_tweets.id',
                    'user.fields': 'username,profile_image_url'
                }
            )
            
            if response and 'data' in response:
                self.last_request_time['timeline'] = datetime.now()
                return response
            else:
                logger.warning(f"No timeline data found: {response}")
                return {'data': []}
                
        except Exception as e:
            logger.error(f"Error getting timeline: {str(e)}")
            return {'data': []}

    async def _wait_for_rate_limit(self, action_type: str):
        """Wait if needed to respect rate limits"""
        if action_type in self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time[action_type]).total_seconds()
            if elapsed < self.min_request_interval[action_type]:
                wait_time = self.min_request_interval[action_type] - elapsed
                await asyncio.sleep(wait_time)

    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """Make a rate-limited request to the Twitter API"""
        if not self.auth_headers:
            raise ValueError("No authentication headers available")
            
        url = f"{self.api_base}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, headers=self.auth_headers, params=params) as response:
                    if response.status == 429:  # Too Many Requests
                        retry_after = int(response.headers.get('Retry-After', '60'))
                        logger.warning(f"Rate limited. Waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        return await self._make_request(method, endpoint, params, data)
                    
                    response_text = await response.text()
                    if not response_text.strip():
                        logger.warning("Empty response from Twitter API")
                        return {'data': []}
                        
                    try:
                        json_response = json.loads(response_text)
                        if response.status != 200:
                            logger.error(f"API error: {json_response}")
                            return {'data': []}
                        return json_response
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON response: {response_text}")
                        return {'data': []}
                        
            elif method == "POST":
                async with session.post(url, headers=self.auth_headers, json=data) as response:
                    if response.status == 429:  # Too Many Requests
                        retry_after = int(response.headers.get('Retry-After', '60'))
                        logger.warning(f"Rate limited. Waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        return await self._make_request(method, endpoint, params, data)
                    
                    response_text = await response.text()
                    if not response_text.strip():
                        logger.warning("Empty response from Twitter API")
                        return {'data': []}
                        
                    try:
                        json_response = json.loads(response_text)
                        if response.status != 200 and response.status != 201:
                            logger.error(f"API error: {json_response}")
                            return {'data': []}
                        return json_response
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON response: {response_text}")
                        return {'data': []}
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user information by ID"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(
                    f"{self.api_base}/users/{user_id}",
                    headers=self.auth_headers,
                    params={
                        'user.fields': 'description,public_metrics,created_at,profile_image_url'
                    }
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    if self._handle_rate_limit(response):
                        retry_count += 1
                        continue
                else:
                    logger.error(f"Error getting user info: {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error getting user info: {str(e)}")
                return None
            retry_count += 1
        return None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user information by username"""
        try:
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
                logger.error(f"Error getting user info: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting user info: {str(e)}")
            return None

    def get_home_timeline(self, count: int = 10) -> List[Dict]:
        """
        Get home timeline using Bearer token authentication
        Returns a list of tweet dictionaries
        """
        def create_url():
            if not self.user_id:
                raise ValueError("User ID not found - check authentication")
            return f"https://api.twitter.com/2/users/{self.user_id}/timelines/reverse_chronological"

        def get_params():
            return {
                "max_results": min(count, 100),  # Twitter v2 API max is 100
                "tweet.fields": "created_at,public_metrics,conversation_id,referenced_tweets",
                "user.fields": "name,username,description",
                "expansions": "author_id,referenced_tweets.id,in_reply_to_user_id"
            }

        def bearer_oauth(r):
            """Method required by bearer token authentication."""
            r.headers["Authorization"] = f"Bearer {self.auth_headers['Authorization'].split(' ')[1]}"
            r.headers["User-Agent"] = "DeepYearner/1.0"
            return r

        try:
            url = create_url()
            params = get_params()
            
            response = requests.request(
                "GET", 
                url, 
                auth=bearer_oauth,
                params=params
            )
            
            if response.status_code != 200:
                logger.error(
                    f"Request returned an error: {response.status_code} {response.text}"
                )
                return []

            # Parse the response
            json_response = response.json()
            
            # Convert to our standard format
            timeline = []
            if 'data' in json_response:
                users = {u['id']: u for u in json_response.get('includes', {}).get('users', [])}
                referenced_tweets = {t['id']: t for t in json_response.get('includes', {}).get('tweets', [])}
                
                for tweet in json_response['data']:
                    user = users.get(tweet['author_id'], {})
                    timeline.append({
                        'id': tweet['id'],
                        'text': tweet['text'],
                        'created_at': tweet['created_at'],
                        'user': {
                            'id': tweet['author_id'],
                            'screen_name': user.get('username', 'unknown'),
                            'name': user.get('name', 'unknown'),
                            'description': user.get('description', '')
                        },
                        'metrics': tweet.get('public_metrics', {}),
                        'conversation_id': tweet.get('conversation_id'),
                        'referenced_tweets': [
                            {
                                'type': ref['type'],
                                'text': referenced_tweets.get(ref['id'], {}).get('text', ''),
                                'author_id': referenced_tweets.get(ref['id'], {}).get('author_id')
                            }
                            for ref in tweet.get('referenced_tweets', [])
                        ],
                        'in_reply_to_user_id': tweet.get('in_reply_to_user_id')
                    })
            
            return timeline

        except Exception as e:
            logger.error(f"Error getting timeline: {str(e)}")
            return []

def get_twitter_client() -> TwitterClient:
    """Factory function to create TwitterClient instance"""
    return TwitterClient()
