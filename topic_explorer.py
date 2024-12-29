from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import random
import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import re

from twitter_utils import TwitterAPI
from prompt_manager import PromptManager

logger = logging.getLogger(__name__)

@dataclass
class TopicEngagementMetrics:
    """Metrics for topic engagement success"""
    likes: int = 0
    replies: int = 0
    retweets: int = 0
    quotes: int = 0
    sentiment_score: float = 0.0
    
    def meets_criteria(self, criteria: Dict) -> bool:
        """Check if metrics meet success criteria"""
        return (
            self.likes >= criteria.get('min_likes', 0) and
            self.replies >= criteria.get('min_replies', 0) and
            self.sentiment_score >= criteria.get('sentiment_threshold', 0.0)
        )

class TopicExplorer:
    def __init__(
        self,
        twitter_api: TwitterAPI,
        personality_state: Any,
        db: Any,
        vector_store: Any
    ):
        self.twitter_api = twitter_api
        self.personality_state = personality_state
        self.db = db
        self.vector_store = vector_store
        self.prompt_manager = PromptManager()
        
    async def discover_and_engage(self) -> None:
        """Main loop for topic discovery and engagement"""
        # Get discovery plan
        discovery_plan = await self._get_discovery_plan()
        if not discovery_plan:
            logger.error("Failed to get discovery plan")
            return
            
        # Execute plan
        for query_idx in discovery_plan['priority_order']:
            query_info = discovery_plan['search_queries'][query_idx]
            await self._execute_search_query(query_info)
            
            # Respect rate limits and exploration duration
            await asyncio.sleep(
                discovery_plan['exploration_duration_minutes'] * 60 / 
                len(discovery_plan['priority_order'])
            )
            
    async def _get_discovery_plan(self) -> Dict:
        """Get topic discovery plan from Claude"""
        try:
            # Get context for prompt
            context = {
                "personality_state": self.personality_state.get_state_summary(),
                "recent_interactions": self._get_recent_interactions(),
                "active_bits": self._get_active_bits(),
                "recent_topics": self._get_recent_topics(),
                "timeline_vibe": self._get_timeline_vibe(),
                "current_time": datetime.now().strftime("%H:%M"),
                "day_of_week": datetime.now().strftime("%A")
            }
            
            # Get and format prompt
            prompt = self.prompt_manager.load_prompt("topic_discovery.txt")
            formatted_prompt = prompt.format(**context)
            
            # Get response from Claude
            response = await self.personality_state.get_response(formatted_prompt)
            
            # Extract plan
            plan_match = re.search(
                r'<discovery_plan>(.*?)</discovery_plan>', 
                response, 
                re.DOTALL
            )
            if plan_match:
                return json.loads(plan_match.group(1))
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting discovery plan: {str(e)}")
            return None
            
    async def _execute_search_query(self, query_info: Dict) -> None:
        """Execute a single search query and engage with results"""
        try:
            # Search for tweets
            tweets = self.twitter_api.search_recent_tweets(
                query_info['query'],
                max_results=100
            )
            
            # Filter and sort tweets
            relevant_tweets = self._filter_tweets(tweets, query_info)
            
            # Engage with top tweets
            engagements = 0
            for tweet in relevant_tweets:
                if engagements >= query_info['max_engagements_per_query']:
                    break
                    
                if await self._engage_with_tweet(tweet, query_info):
                    engagements += 1
                    
        except Exception as e:
            logger.error(f"Error executing query {query_info['query']}: {str(e)}")
            
    def _filter_tweets(self, tweets: List[Dict], query_info: Dict) -> List[Dict]:
        """Filter and sort tweets based on criteria"""
        filtered = []
        for tweet in tweets:
            # Skip if already engaged
            if self._have_engaged(tweet['id']):
                continue
                
            # Get engagement metrics
            metrics = self._get_tweet_metrics(tweet)
            
            # Check if meets criteria
            if metrics.meets_criteria(query_info['success_metrics']):
                filtered.append({
                    'tweet': tweet,
                    'metrics': metrics
                })
                
        # Sort by metrics
        filtered.sort(
            key=lambda x: (
                x['metrics'].likes + 
                x['metrics'].replies * 2 + 
                x['metrics'].retweets * 1.5
            ),
            reverse=True
        )
        
        return [f['tweet'] for f in filtered]
        
    async def _engage_with_tweet(self, tweet: Dict, query_info: Dict) -> bool:
        """Engage with a single tweet based on query info"""
        try:
            engagement_type = query_info['engagement_type']
            
            # Get engagement response if needed
            response = None
            if engagement_type in ['reply', 'quote']:
                response = await self._get_engagement_response(tweet, query_info)
                if not response:
                    return False
            
            # Execute engagement
            if engagement_type == 'like':
                success = self.twitter_api.like_tweet(tweet['id'])
            elif engagement_type == 'reply':
                success = bool(self.twitter_api.reply_to_tweet(
                    tweet['id'],
                    response
                ))
            elif engagement_type == 'retweet':
                success = self.twitter_api.retweet(tweet['id'])
            elif engagement_type == 'quote':
                success = bool(self.twitter_api.quote_tweet(
                    tweet['id'],
                    response
                ))
            else:
                return False
                
            # Record engagement
            if success:
                self._record_engagement(tweet, query_info, engagement_type)
                
            return success
            
        except Exception as e:
            logger.error(f"Error engaging with tweet {tweet['id']}: {str(e)}")
            return False
            
    async def _get_engagement_response(
        self,
        tweet: Dict,
        query_info: Dict
    ) -> Optional[str]:
        """Get an engaging response for reply or quote tweet with personality alignment"""
        try:
            # Get conversation context
            conversation = self.twitter_api.get_conversation_thread(tweet['id'])
            
            # Get author info and mutual history
            author = self.twitter_api.get_user_by_id(tweet['author_id'])
            mutual_history = self._get_mutual_history(tweet['author_id'])
            
            # Get timeline analysis
            timeline_analysis = await self._analyze_timeline()
            
            # Get topic categorization
            topic_category = self._categorize_topic(tweet['text'], query_info['query'])
            
            # Prepare rich context for prompt
            context = {
                "author": {
                    "username": author.username,
                    "name": author.name,
                    "followers": author.public_metrics['followers_count'],
                    "is_mutual": bool(mutual_history)
                },
                "tweet_text": tweet['text'],
                "created_at": tweet['created_at'],
                "like_count": tweet['public_metrics'].get('like_count', 0),
                "reply_count": tweet['public_metrics'].get('reply_count', 0),
                "retweet_count": tweet['public_metrics'].get('retweet_count', 0),
                "conversation_context": self._format_conversation(conversation),
                "personality_mode": self.personality_state.current_mode,
                "energy_level": self.personality_state.energy_level,
                "recent_vibes": self.personality_state.recent_vibes,
                "active_bits": self._get_active_bits(),
                "mutual_history": mutual_history,
                "engagement_type": query_info['engagement_type'],
                "intent": query_info['intent'],
                "topic_category": topic_category,
                "timeline_vibe": timeline_analysis['vibe'],
                "trending_topics": timeline_analysis['trending_topics'],
                "network_mood": timeline_analysis['network_mood']
            }
            
            # Get response from Claude
            prompt = self.prompt_manager.load_prompt("engagement_response.txt")
            formatted_prompt = prompt.format(**context)
            
            response = await self.personality_state.get_response(formatted_prompt)
            
            # Extract and validate response
            response_data = self._extract_response_data(response)
            if not response_data:
                return None
                
            # Validate response meets criteria
            if not self._validate_response(response_data, context):
                logger.warning("Response failed validation criteria")
                return None
                
            # Store response metadata for learning
            self._store_response_metadata(tweet['id'], response_data)
            
            return response_data['response_text']
            
        except Exception as e:
            logger.error(f"Error generating engagement response: {str(e)}")
            return None
            
    def _format_conversation(self, conversation: List[Dict]) -> str:
        """Format conversation thread for context"""
        formatted = []
        for tweet in conversation:
            author = self.twitter_api.get_user_by_id(tweet['author_id'])
            formatted.append(f"@{author.username}: {tweet['text']}")
        return "\n".join(formatted)
        
    def _get_mutual_history(self, author_id: str) -> List[Dict]:
        """Get history of mutual interactions with author"""
        return list(self.db["interactions"].rows_where(
            "author_id = ? AND timestamp > datetime('now', '-30 days')",
            [author_id],
            order_by="timestamp desc",
            limit=10
        ))
        
    async def _analyze_timeline(self) -> Dict:
        """Analyze current timeline for context"""
        # Get recent timeline tweets
        timeline = self.twitter_api.get_user_timeline(count=100)
        
        # Analyze engagement patterns
        engagement_patterns = self._analyze_engagement_patterns(timeline)
        
        # Get trending topics
        trending = self._extract_trending_topics(timeline)
        
        # Analyze network mood
        mood = await self._analyze_network_mood(timeline)
        
        return {
            "vibe": engagement_patterns,
            "trending_topics": trending,
            "network_mood": mood
        }
        
    def _categorize_topic(self, tweet_text: str, query: str) -> str:
        """Categorize the topic of the tweet"""
        # TODO: Implement more sophisticated topic categorization
        categories = [
            "tech", "politics", "culture", "science", 
            "entertainment", "sports", "memes"
        ]
        return random.choice(categories)
        
    def _extract_response_data(self, response: str) -> Optional[Dict]:
        """Extract structured response data from Claude's response"""
        try:
            response_match = re.search(
                r'<engagement_response>(.*?)</engagement_response>',
                response,
                re.DOTALL
            )
            if response_match:
                return json.loads(response_match.group(1))
            return None
        except Exception as e:
            logger.error(f"Error extracting response data: {str(e)}")
            return None
            
    def _validate_response(self, response_data: Dict, context: Dict) -> bool:
        """Validate response meets quality criteria"""
        try:
            # Check confidence threshold
            if response_data['confidence_score'] < 0.7:
                return False
                
            # Check personality alignment
            personality_scores = response_data['personality_alignment']
            if any(score < 0.6 for score in personality_scores.values()):
                return False
                
            # Check risk assessment
            risk = response_data['risk_assessment']
            if risk['controversy_level'] > 0.7 or risk['potential_misinterpretation'] > 0.6:
                return False
                
            # Validate response length
            if len(response_data['response_text']) > 280:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            return False
            
    def _store_response_metadata(self, tweet_id: str, response_data: Dict) -> None:
        """Store response metadata for learning"""
        self.db["response_metadata"].insert({
            "tweet_id": tweet_id,
            "personality_alignment": json.dumps(response_data['personality_alignment']),
            "strategy": json.dumps(response_data['strategy']),
            "risk_assessment": json.dumps(response_data['risk_assessment']),
            "metadata": json.dumps(response_data['metadata']),
            "timestamp": datetime.now().isoformat()
        })
        
    def _analyze_engagement_patterns(self, timeline: List[Dict]) -> Dict:
        """Analyze engagement patterns in timeline"""
        total_engagement = 0
        peak_hours = {}
        
        for tweet in timeline:
            # Calculate total engagement
            metrics = tweet['public_metrics']
            engagement = (
                metrics.get('like_count', 0) +
                metrics.get('reply_count', 0) * 2 +
                metrics.get('retweet_count', 0) * 1.5
            )
            total_engagement += engagement
            
            # Track peak hours
            hour = datetime.fromisoformat(tweet['created_at']).hour
            peak_hours[hour] = peak_hours.get(hour, 0) + engagement
            
        return {
            "total_engagement": total_engagement,
            "peak_hours": peak_hours,
            "avg_engagement": total_engagement / len(timeline) if timeline else 0
        }
        
    def _extract_trending_topics(self, timeline: List[Dict]) -> List[str]:
        """Extract trending topics from timeline"""
        # TODO: Implement more sophisticated topic extraction
        topics = {}
        for tweet in timeline:
            # Extract hashtags
            hashtags = re.findall(r'#(\w+)', tweet['text'])
            for tag in hashtags:
                topics[tag] = topics.get(tag, 0) + 1
                
        # Return top 5 topics
        return sorted(
            topics.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
    async def _analyze_network_mood(self, timeline: List[Dict]) -> Dict:
        """Analyze the mood of the network"""
        # TODO: Implement more sophisticated mood analysis
        return {
            "overall_sentiment": 0.7,
            "energy_level": "high",
            "conversation_types": ["casual", "technical", "humorous"]
        }
        
    def _get_tweet_metrics(self, tweet: Dict) -> TopicEngagementMetrics:
        """Get engagement metrics for a tweet"""
        metrics = tweet.get('public_metrics', {})
        return TopicEngagementMetrics(
            likes=metrics.get('like_count', 0),
            replies=metrics.get('reply_count', 0),
            retweets=metrics.get('retweet_count', 0),
            quotes=metrics.get('quote_count', 0),
            sentiment_score=self._analyze_sentiment(tweet)
        )
        
    def _analyze_sentiment(self, tweet: Dict) -> float:
        """Analyze sentiment of tweet and replies"""
        # TODO: Implement sentiment analysis
        return 0.7
        
    def _have_engaged(self, tweet_id: str) -> bool:
        """Check if we've already engaged with this tweet"""
        return bool(self.db["interactions"].find_one(tweet_id=tweet_id))
        
    def _record_engagement(
        self,
        tweet: Dict,
        query_info: Dict,
        engagement_type: str
    ) -> None:
        """Record an engagement in the database"""
        self.db["interactions"].insert({
            "tweet_id": tweet['id'],
            "author_id": tweet['author_id'],
            "query": query_info['query'],
            "intent": query_info['intent'],
            "engagement_type": engagement_type,
            "timestamp": datetime.now().isoformat()
        })
        
    def _get_recent_interactions(self) -> List[Dict]:
        """Get recent interactions from database"""
        return list(self.db["interactions"].rows_where(
            order_by="timestamp desc",
            limit=50
        ))
        
    def _get_active_bits(self) -> List[Dict]:
        """Get active running bits"""
        return list(self.db["meme_tracker"].rows_where(
            "success_rate > ? AND last_used > datetime('now', '-7 days')",
            [0.6],
            order_by="last_used desc",
            limit=10
        ))
        
    def _get_recent_topics(self) -> List[Dict]:
        """Get recently successful topics"""
        topics = {}
        for interaction in self.db["interactions"].rows_where(
            "timestamp > datetime('now', '-7 days')"
        ):
            query = interaction['query']
            if query not in topics:
                topics[query] = {
                    'count': 0,
                    'success': 0
                }
            topics[query]['count'] += 1
            if interaction.get('success', True):
                topics[query]['success'] += 1
                
        return [
            {
                'query': query,
                'success_rate': data['success'] / data['count']
            }
            for query, data in topics.items()
            if data['count'] >= 3 and data['success'] / data['count'] >= 0.6
        ]
        
    def _get_timeline_vibe(self) -> Dict:
        """Get current timeline vibe analysis"""
        # TODO: Implement timeline vibe analysis
        return {
            "mood": "energetic",
            "trending_topics": ["tech", "memes", "coding"],
            "sentiment": 0.8
        }
