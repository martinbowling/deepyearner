import tweepy
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
import time
import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TwitterCredentials:
    """Container for Twitter API credentials"""
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str
    bearer_token: str

class RateLimitHandler:
    """Handles Twitter API rate limits"""
    def __init__(self):
        self.rate_limits = {}
        
    def handle_rate_limit(self, endpoint: str):
        """Sleep if rate limit is hit"""
        if endpoint in self.rate_limits:
            time_to_reset = self.rate_limits[endpoint] - time.time()
            if time_to_reset > 0:
                logger.info(f"Rate limit hit for {endpoint}. Sleeping for {time_to_reset} seconds")
                time.sleep(time_to_reset)
                
    def update_rate_limit(self, endpoint: str, reset_time: float):
        """Update rate limit info for endpoint"""
        self.rate_limits[endpoint] = reset_time

class TwitterAPI:
    """Wrapper for Twitter API functionality"""
    
    def __init__(self, credentials: TwitterCredentials):
        """Initialize with Twitter credentials"""
        self.client = tweepy.Client(
            bearer_token=credentials.bearer_token,
            consumer_key=credentials.consumer_key,
            consumer_secret=credentials.consumer_secret,
            access_token=credentials.access_token,
            access_token_secret=credentials.access_token_secret,
            wait_on_rate_limit=True
        )
        self.rate_limiter = RateLimitHandler()
        
    def create_tweet(
        self,
        text: str,
        reply_to: Optional[str] = None,
        quote_tweet: Optional[str] = None,
        media_ids: Optional[List[str]] = None
    ) -> Dict:
        """Create a new tweet"""
        try:
            response = self.client.create_tweet(
                text=text,
                in_reply_to_tweet_id=reply_to,
                quote_tweet_id=quote_tweet,
                media_ids=media_ids
            )
            return {
                'id': response.data['id'],
                'text': text,
                'created_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating tweet: {str(e)}")
            return None
            
    def reply_to_tweet(
        self,
        tweet_id: str,
        text: str,
        media_ids: Optional[List[str]] = None
    ) -> Dict:
        """Reply to a specific tweet"""
        return self.create_tweet(text=text, reply_to=tweet_id, media_ids=media_ids)
        
    def quote_tweet(
        self,
        tweet_id: str,
        text: str,
        media_ids: Optional[List[str]] = None
    ) -> Dict:
        """Quote tweet another tweet"""
        return self.create_tweet(text=text, quote_tweet=tweet_id, media_ids=media_ids)
        
    def get_tweet(self, tweet_id: str) -> Dict:
        """Get a single tweet by ID"""
        try:
            tweet = self.client.get_tweet(
                tweet_id,
                expansions=['author_id', 'referenced_tweets.id'],
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            )
            return tweet.data
        except Exception as e:
            logger.error(f"Error getting tweet {tweet_id}: {str(e)}")
            return None
            
    def get_user_tweets(
        self,
        user_id: str,
        max_results: int = 100,
        since_id: Optional[str] = None
    ) -> List[Dict]:
        """Get tweets from a specific user"""
        try:
            tweets = []
            for response in tweepy.Paginator(
                self.client.get_users_tweets,
                user_id,
                max_results=max_results,
                since_id=since_id,
                tweet_fields=['created_at', 'public_metrics']
            ):
                if response.data:
                    tweets.extend(response.data)
            return tweets
        except Exception as e:
            logger.error(f"Error getting user tweets for {user_id}: {str(e)}")
            return []
            
    def search_recent_tweets(
        self,
        query: str,
        max_results: int = 100,
        since_id: Optional[str] = None
    ) -> List[Dict]:
        """Search for recent tweets matching query"""
        try:
            tweets = []
            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=max_results,
                since_id=since_id,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            ):
                if response.data:
                    tweets.extend(response.data)
            return tweets
        except Exception as e:
            logger.error(f"Error searching tweets for {query}: {str(e)}")
            return []
            
    def get_user_timeline(
        self,
        user_id: Optional[str] = None,
        screen_name: Optional[str] = None,
        count: int = 200
    ) -> List[Dict]:
        """Get user's timeline tweets"""
        try:
            if not user_id and not screen_name:
                raise ValueError("Must provide either user_id or screen_name")
                
            tweets = self.client.get_users_tweets(
                id=user_id,
                username=screen_name,
                max_results=count,
                tweet_fields=['created_at', 'public_metrics']
            )
            return tweets.data if tweets.data else []
        except Exception as e:
            logger.error(f"Error getting timeline: {str(e)}")
            return []
            
    def get_user_followers(
        self,
        user_id: str,
        max_results: int = 1000
    ) -> List[Dict]:
        """Get user's followers"""
        try:
            followers = []
            for response in tweepy.Paginator(
                self.client.get_users_followers,
                user_id,
                max_results=max_results,
                user_fields=['description', 'public_metrics']
            ):
                if response.data:
                    followers.extend(response.data)
            return followers
        except Exception as e:
            logger.error(f"Error getting followers for {user_id}: {str(e)}")
            return []
            
    def get_user_following(
        self,
        user_id: str,
        max_results: int = 1000
    ) -> List[Dict]:
        """Get users that the specified user is following"""
        try:
            following = []
            for response in tweepy.Paginator(
                self.client.get_users_following,
                user_id,
                max_results=max_results,
                user_fields=['description', 'public_metrics']
            ):
                if response.data:
                    following.extend(response.data)
            return following
        except Exception as e:
            logger.error(f"Error getting following for {user_id}: {str(e)}")
            return []
            
    def get_tweet_liking_users(
        self,
        tweet_id: str,
        max_results: int = 100
    ) -> List[Dict]:
        """Get users who liked a tweet"""
        try:
            liking_users = []
            for response in tweepy.Paginator(
                self.client.get_liking_users,
                tweet_id,
                max_results=max_results,
                user_fields=['description', 'public_metrics']
            ):
                if response.data:
                    liking_users.extend(response.data)
            return liking_users
        except Exception as e:
            logger.error(f"Error getting liking users for tweet {tweet_id}: {str(e)}")
            return []
            
    def get_tweet_retweeters(
        self,
        tweet_id: str,
        max_results: int = 100
    ) -> List[Dict]:
        """Get users who retweeted a tweet"""
        try:
            retweeters = []
            for response in tweepy.Paginator(
                self.client.get_retweeters,
                tweet_id,
                max_results=max_results,
                user_fields=['description', 'public_metrics']
            ):
                if response.data:
                    retweeters.extend(response.data)
            return retweeters
        except Exception as e:
            logger.error(f"Error getting retweeters for tweet {tweet_id}: {str(e)}")
            return []
            
    def get_user_mentions(
        self,
        user_id: str,
        max_results: int = 100,
        since_id: Optional[str] = None
    ) -> List[Dict]:
        """Get tweets mentioning a user"""
        try:
            mentions = []
            for response in tweepy.Paginator(
                self.client.get_users_mentions,
                user_id,
                max_results=max_results,
                since_id=since_id,
                tweet_fields=['created_at', 'public_metrics']
            ):
                if response.data:
                    mentions.extend(response.data)
            return mentions
        except Exception as e:
            logger.error(f"Error getting mentions for {user_id}: {str(e)}")
            return []
            
    def like_tweet(self, tweet_id: str) -> bool:
        """Like a tweet"""
        try:
            self.client.like(tweet_id)
            return True
        except Exception as e:
            logger.error(f"Error liking tweet {tweet_id}: {str(e)}")
            return False
            
    def unlike_tweet(self, tweet_id: str) -> bool:
        """Unlike a tweet"""
        try:
            self.client.unlike(tweet_id)
            return True
        except Exception as e:
            logger.error(f"Error unliking tweet {tweet_id}: {str(e)}")
            return False
            
    def retweet(self, tweet_id: str) -> bool:
        """Retweet a tweet"""
        try:
            response = self.client.retweet(
                tweet_id=tweet_id
            )
            if response and response.data:
                logger.info(f"Successfully retweeted tweet {tweet_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error retweeting tweet {tweet_id}: {str(e)}")
            return False

    def unretweet(self, tweet_id: str) -> bool:
        """Remove a retweet"""
        try:
            response = self.client.unretweet(
                tweet_id=tweet_id
            )
            if response and response.data:
                logger.info(f"Successfully unretweeted tweet {tweet_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error unretweeting tweet {tweet_id}: {str(e)}")
            return False

    def quote_tweet(self, tweet_id: str, text: str) -> Optional[Dict]:
        """Quote tweet with comment"""
        try:
            # Get original tweet URL
            tweet = self.client.get_tweet(tweet_id)
            if not tweet or not tweet.data:
                logger.error(f"Could not find tweet {tweet_id}")
                return None

            # Create quote tweet
            response = self.client.create_tweet(
                text=text,
                quote_tweet_id=tweet_id
            )
            
            if response and response.data:
                logger.info(f"Successfully quote tweeted {tweet_id}")
                return response.data
            return None
        except Exception as e:
            logger.error(f"Error quote tweeting {tweet_id}: {str(e)}")
            return None
            
    def follow_user(self, user_id: str) -> bool:
        """Follow a user"""
        try:
            self.client.follow_user(user_id)
            return True
        except Exception as e:
            logger.error(f"Error following user {user_id}: {str(e)}")
            return False
            
    def unfollow_user(self, user_id: str) -> bool:
        """Unfollow a user"""
        try:
            self.client.unfollow_user(user_id)
            return True
        except Exception as e:
            logger.error(f"Error unfollowing user {user_id}: {str(e)}")
            return False
            
    def get_user_by_username(self, username: str) -> Dict:
        """Get user information by username"""
        try:
            user = self.client.get_user(
                username=username,
                user_fields=['description', 'public_metrics', 'created_at']
            )
            return user.data
        except Exception as e:
            logger.error(f"Error getting user {username}: {str(e)}")
            return None
            
    def get_user_by_id(self, user_id: str) -> Dict:
        """Get user information by ID"""
        try:
            user = self.client.get_user(
                id=user_id,
                user_fields=['description', 'public_metrics', 'created_at']
            )
            return user.data
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {str(e)}")
            return None
            
    def get_conversation_thread(
        self,
        tweet_id: str,
        max_results: int = 100
    ) -> List[Dict]:
        """Get conversation thread for a tweet"""
        try:
            conversation = []
            tweet = self.get_tweet(tweet_id)
            if not tweet:
                return []
                
            # Get replies
            query = f"conversation_id:{tweet_id}"
            replies = self.search_recent_tweets(query, max_results)
            
            # Sort by time
            conversation = [tweet] + replies
            conversation.sort(key=lambda x: x.created_at)
            
            return conversation
        except Exception as e:
            logger.error(f"Error getting conversation for tweet {tweet_id}: {str(e)}")
            return []

def get_twitter_api() -> TwitterAPI:
    """Create TwitterAPI instance from environment variables"""
    credentials = TwitterCredentials(
        consumer_key=os.getenv("TWITTER_CONSUMER_KEY"),
        consumer_secret=os.getenv("TWITTER_CONSUMER_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        bearer_token=os.getenv("TWITTER_BEARER_TOKEN")
    )
    return TwitterAPI(credentials)
