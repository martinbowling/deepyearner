from typing import Dict, List, Optional, Any
import logging
import tweepy
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class TwitterActions:
    """Handles all Twitter posting actions"""
    
    def __init__(self, api: tweepy.API):
        self.api = api
        self.rate_limits = {
            'tweets': {'count': 0, 'reset': datetime.now()},
            'replies': {'count': 0, 'reset': datetime.now()},
            'retweets': {'count': 0, 'reset': datetime.now()},
            'quotes': {'count': 0, 'reset': datetime.now()}
        }
        
    def post_tweet(
        self,
        text: str,
        reply_to: Optional[str] = None,
        quote_tweet: Optional[str] = None,
        media: Optional[List[str]] = None
    ) -> Optional[str]:
        """Post a new tweet"""
        try:
            # Check rate limits
            if not self._check_rate_limit('tweets'):
                logger.warning("Tweet rate limit reached")
                return None
                
            # Prepare tweet parameters
            params = {
                'text': text,
                'in_reply_to_status_id': reply_to
            }
            
            # Add media if provided
            if media:
                media_ids = []
                for media_file in media:
                    uploaded = self.api.media_upload(media_file)
                    media_ids.append(uploaded.media_id)
                params['media_ids'] = media_ids
                
            # Post tweet
            if quote_tweet:
                tweet = self.api.update_status(
                    **params,
                    attachment_url=f"https://twitter.com/i/web/status/{quote_tweet}"
                )
            else:
                tweet = self.api.update_status(**params)
                
            # Update rate limit
            self._update_rate_limit('tweets')
            
            return tweet.id_str
            
        except Exception as e:
            logger.error(f"Error posting tweet: {str(e)}")
            return None
            
    def reply_to_tweet(
        self,
        tweet_id: str,
        text: str,
        quote: bool = False,
        media: Optional[List[str]] = None
    ) -> Optional[str]:
        """Reply to a tweet"""
        try:
            # Check rate limits
            if not self._check_rate_limit('replies'):
                logger.warning("Reply rate limit reached")
                return None
                
            # Get original tweet for context
            original = self.api.get_status(tweet_id)
            
            # Format reply
            if not text.startswith('@'):
                text = f"@{original.user.screen_name} {text}"
                
            # Post reply
            if quote:
                result = self.post_tweet(text, quote_tweet=tweet_id, media=media)
            else:
                result = self.post_tweet(text, reply_to=tweet_id, media=media)
                
            # Update rate limit
            self._update_rate_limit('replies')
            
            return result
            
        except Exception as e:
            logger.error(f"Error replying to tweet: {str(e)}")
            return None
            
    def retweet(self, tweet_id: str) -> bool:
        """Retweet a tweet"""
        try:
            # Check rate limits
            if not self._check_rate_limit('retweets'):
                logger.warning("Retweet rate limit reached")
                return False
                
            # Retweet
            self.api.retweet(tweet_id)
            
            # Update rate limit
            self._update_rate_limit('retweets')
            
            return True
            
        except Exception as e:
            logger.error(f"Error retweeting: {str(e)}")
            return False
            
    def quote_tweet(
        self,
        tweet_id: str,
        text: str,
        media: Optional[List[str]] = None
    ) -> Optional[str]:
        """Quote tweet with comment"""
        try:
            # Check rate limits
            if not self._check_rate_limit('quotes'):
                logger.warning("Quote tweet rate limit reached")
                return None
                
            # Post quote tweet
            result = self.post_tweet(text, quote_tweet=tweet_id, media=media)
            
            # Update rate limit
            self._update_rate_limit('quotes')
            
            return result
            
        except Exception as e:
            logger.error(f"Error quote tweeting: {str(e)}")
            return None
            
    def post_thread(
        self,
        tweets: List[str],
        reply_to: Optional[str] = None,
        media: Optional[List[List[str]]] = None
    ) -> List[str]:
        """Post a thread of tweets"""
        thread_ids = []
        current_reply_to = reply_to
        
        try:
            for i, tweet in enumerate(tweets):
                # Get media for this tweet if provided
                tweet_media = media[i] if media and i < len(media) else None
                
                # Post tweet
                tweet_id = self.post_tweet(
                    tweet,
                    reply_to=current_reply_to,
                    media=tweet_media
                )
                
                if tweet_id:
                    thread_ids.append(tweet_id)
                    current_reply_to = tweet_id
                else:
                    logger.error(f"Failed to post tweet {i} in thread")
                    break
                    
            return thread_ids
            
        except Exception as e:
            logger.error(f"Error posting thread: {str(e)}")
            return thread_ids
            
    def _check_rate_limit(self, action_type: str) -> bool:
        """Check if we're within rate limits"""
        limit = self.rate_limits[action_type]
        now = datetime.now()
        
        # Reset if needed
        if now > limit['reset']:
            limit['count'] = 0
            limit['reset'] = now.replace(hour=now.hour + 1, minute=0, second=0)
            
        # Check limit
        max_per_hour = {
            'tweets': 300,
            'replies': 300,
            'retweets': 300,
            'quotes': 300
        }
        
        return limit['count'] < max_per_hour[action_type]
        
    def _update_rate_limit(self, action_type: str):
        """Update rate limit counter"""
        self.rate_limits[action_type]['count'] += 1
