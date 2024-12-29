from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from datetime import datetime, timedelta
import json
import random
from collections import defaultdict

from twitter_utils import TwitterAPI
from prompt_manager import PromptManager
from personality_state import PersonalityState

logger = logging.getLogger(__name__)

class ContentPattern:
    """Tracks successful content patterns"""
    def __init__(self, pattern_type: str):
        self.pattern_type = pattern_type
        self.examples: List[Dict] = []
        self.success_rate = 0.0
        self.last_used = datetime.now()
        self.engagement_stats = {
            'avg_likes': 0.0,
            'avg_replies': 0.0,
            'avg_retweets': 0.0
        }
        
    def update_stats(self, new_example: Dict):
        """Update pattern stats with new example"""
        self.examples.append(new_example)
        self.last_used = datetime.now()
        
        # Update engagement averages
        total_examples = len(self.examples)
        self.engagement_stats = {
            'avg_likes': sum(e['metrics']['likes'] for e in self.examples) / total_examples,
            'avg_replies': sum(e['metrics']['replies'] for e in self.examples) / total_examples,
            'avg_retweets': sum(e['metrics']['retweets'] for e in self.examples) / total_examples
        }
        
        # Update success rate
        successes = sum(1 for e in self.examples if e['metrics']['success'])
        self.success_rate = successes / total_examples

class ContentDiscovery:
    """Discovers and analyzes content opportunities"""
    
    def __init__(
        self,
        twitter_api: TwitterAPI,
        personality: PersonalityState,
        prompt_manager: PromptManager,
        db: Any,
        vector_store: Any
    ):
        self.twitter_api = twitter_api
        self.personality = personality
        self.prompt_manager = prompt_manager
        self.db = db
        self.vector_store = vector_store
        self.content_patterns: Dict[str, ContentPattern] = {}
        self.trending_cache = {}
        self.trending_cache_time = None
        self.timeline_cursor = None
        
    async def discover_content(self) -> List[Dict]:
        """Main content discovery process"""
        discoveries = []
        
        # 1. Check trending topics
        trending = await self.get_trending_topics()
        if trending:
            trend_opportunities = await self.analyze_trends(trending)
            discoveries.extend(trend_opportunities)
        
        # 2. Browse timeline
        timeline_content = await self.browse_timeline()
        if timeline_content:
            timeline_opportunities = await self.analyze_timeline_content(timeline_content)
            discoveries.extend(timeline_opportunities)
        
        # 3. Search interesting topics
        search_results = await self.search_topics()
        if search_results:
            search_opportunities = await self.analyze_search_results(search_results)
            discoveries.extend(search_opportunities)
        
        # 4. Update content patterns
        self.update_patterns(discoveries)
        
        return discoveries
        
    async def get_trending_topics(self) -> List[Dict]:
        """Get and cache trending topics"""
        try:
            now = datetime.now()
            
            # Return cached trends if recent
            if (self.trending_cache_time and 
                now - self.trending_cache_time < timedelta(minutes=15)):
                return self.trending_cache
                
            # Get fresh trends
            trends = self.twitter_api.get_trending_topics()
            
            # Process and store trends
            processed_trends = []
            for trend in trends:
                # Get sample tweets for each trend
                sample_tweets = self.twitter_api.search_recent_tweets(
                    trend['query'],
                    max_results=10
                )
                
                processed_trends.append({
                    'name': trend['name'],
                    'query': trend['query'],
                    'tweet_volume': trend['tweet_volume'],
                    'sample_tweets': sample_tweets
                })
            
            # Update cache
            self.trending_cache = processed_trends
            self.trending_cache_time = now
            
            return processed_trends
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {str(e)}")
            return []
            
    async def browse_timeline(self) -> List[Dict]:
        """Browse timeline for interesting content"""
        try:
            # Get timeline with pagination
            timeline = self.twitter_api.get_home_timeline(
                count=50,
                cursor=self.timeline_cursor
            )
            
            if timeline:
                self.timeline_cursor = timeline[-1]['id']
                
            # Group tweets by conversation
            conversations = self.group_conversations(timeline)
            
            # Get engagement metrics
            for tweet in timeline:
                metrics = self.twitter_api.get_tweet_metrics(tweet['id'])
                tweet['metrics'] = metrics
                
            return timeline
            
        except Exception as e:
            logger.error(f"Error browsing timeline: {str(e)}")
            return []
            
    async def search_topics(self) -> List[Dict]:
        """Search for tweets on interesting topics"""
        try:
            # Get search topics from prompt
            topics = await self.get_search_topics()
            
            results = []
            for topic in topics:
                # Search for tweets
                tweets = self.twitter_api.search_recent_tweets(
                    topic['query'],
                    max_results=topic['max_results']
                )
                
                # Get metrics
                for tweet in tweets:
                    metrics = self.twitter_api.get_tweet_metrics(tweet['id'])
                    tweet['metrics'] = metrics
                    tweet['topic'] = topic
                    
                results.extend(tweets)
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching topics: {str(e)}")
            return []
            
    async def analyze_trends(self, trends: List[Dict]) -> List[Dict]:
        """Analyze trending topics for engagement opportunities"""
        opportunities = []
        
        for trend in trends:
            try:
                # Get analysis prompt
                prompt = self.prompt_manager.get_contextualized_prompt(
                    "trend_analysis",
                    self.personality,
                    {
                        'trend': trend,
                        'sample_tweets': trend['sample_tweets']
                    }
                )
                
                # Get analysis from Claude
                response = await self.personality.get_response(prompt)
                analysis = json.loads(response)
                
                if analysis['should_engage']:
                    opportunities.append({
                        'type': 'trend',
                        'trend': trend,
                        'analysis': analysis,
                        'confidence': analysis['confidence_score']
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing trend {trend['name']}: {str(e)}")
                
        return opportunities
        
    async def analyze_timeline_content(self, tweets: List[Dict]) -> List[Dict]:
        """Analyze timeline content for engagement opportunities"""
        opportunities = []
        
        # Group tweets by conversation
        conversations = self.group_conversations(tweets)
        
        for conv_id, conv_tweets in conversations.items():
            try:
                # Get conversation analysis
                prompt = self.prompt_manager.get_contextualized_prompt(
                    "conversation_analysis",
                    self.personality,
                    {'conversation': conv_tweets}
                )
                
                response = await self.personality.get_response(prompt)
                analysis = json.loads(response)
                
                if analysis['should_engage']:
                    opportunities.append({
                        'type': 'conversation',
                        'conversation': conv_tweets,
                        'analysis': analysis,
                        'confidence': analysis['confidence_score']
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing conversation {conv_id}: {str(e)}")
                
        return opportunities
        
    async def analyze_search_results(self, tweets: List[Dict]) -> List[Dict]:
        """Analyze search results for engagement opportunities"""
        opportunities = []
        
        for tweet in tweets:
            try:
                # Get tweet analysis
                prompt = self.prompt_manager.get_contextualized_prompt(
                    "tweet_analysis",
                    self.personality,
                    {
                        'tweet': tweet,
                        'topic': tweet['topic']
                    }
                )
                
                response = await self.personality.get_response(prompt)
                analysis = json.loads(response)
                
                if analysis['should_engage']:
                    opportunities.append({
                        'type': 'search_result',
                        'tweet': tweet,
                        'analysis': analysis,
                        'confidence': analysis['confidence_score']
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing tweet {tweet['id']}: {str(e)}")
                
        return opportunities
        
    async def get_search_topics(self) -> List[Dict]:
        """Get topics to search for based on personality and context"""
        try:
            # Get topic discovery prompt
            prompt = self.prompt_manager.get_contextualized_prompt(
                "topic_discovery",
                self.personality,
                {
                    'recent_patterns': self.get_recent_patterns(),
                    'active_bits': self.get_active_bits()
                }
            )
            
            # Get topics from Claude
            response = await self.personality.get_response(prompt)
            topics = json.loads(response)
            
            return topics['search_queries']
            
        except Exception as e:
            logger.error(f"Error getting search topics: {str(e)}")
            return []
            
    def group_conversations(self, tweets: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tweets by conversation thread"""
        conversations = defaultdict(list)
        
        for tweet in tweets:
            conv_id = tweet.get('conversation_id', tweet['id'])
            conversations[conv_id].append(tweet)
            
        # Sort tweets in each conversation
        for conv_id in conversations:
            conversations[conv_id].sort(key=lambda x: x['created_at'])
            
        return conversations
        
    def update_patterns(self, discoveries: List[Dict]):
        """Update content patterns based on discoveries"""
        for discovery in discoveries:
            pattern_type = f"{discovery['type']}_{discovery['analysis']['approach']}"
            
            if pattern_type not in self.content_patterns:
                self.content_patterns[pattern_type] = ContentPattern(pattern_type)
                
            self.content_patterns[pattern_type].update_stats({
                'discovery': discovery,
                'metrics': {
                    'likes': discovery['analysis'].get('expected_likes', 0),
                    'replies': discovery['analysis'].get('expected_replies', 0),
                    'retweets': discovery['analysis'].get('expected_retweets', 0),
                    'success': discovery['confidence'] > 0.7
                }
            })
            
    def get_recent_patterns(self) -> List[Dict]:
        """Get recently successful content patterns"""
        recent_patterns = []
        
        for pattern in self.content_patterns.values():
            if (datetime.now() - pattern.last_used < timedelta(days=7) and
                pattern.success_rate > 0.6):
                recent_patterns.append({
                    'type': pattern.pattern_type,
                    'success_rate': pattern.success_rate,
                    'engagement_stats': pattern.engagement_stats,
                    'examples': pattern.examples[-3:]  # Last 3 examples
                })
                
        return recent_patterns
        
    def get_active_bits(self) -> List[Dict]:
        """Get currently active memes and bits"""
        return list(self.db["meme_tracker"].rows_where(
            "success_rate > ? AND last_used > datetime('now', '-7 days')",
            [0.6],
            order_by="last_used desc",
            limit=5
        ))
