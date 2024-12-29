from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import random
from collections import defaultdict
import asyncio
from memory_system import MemorySystem, Episode
from content_processor import ProcessedContent
from personality_system import PersonalitySystem

logger = logging.getLogger(__name__)

@dataclass
class ContentSuggestion:
    """Suggested content for tweeting"""
    tweet_text: str
    source_urls: List[str]
    topics: List[str]
    confidence: float
    reasoning: str
    content_type: str  # original, insight, commentary, etc.
    reference_content: List[Dict]  # Referenced content from memory
    personality_alignment: float
    timing_score: float
    expected_engagement: float

class ContentGenerator:
    """Generates tweet content using memory and personality"""
    
    def __init__(
        self,
        memory_system: MemorySystem,
        personality: PersonalitySystem,
        anthropic_client: Any
    ):
        self.memory = memory_system
        self.personality = personality
        self.client = anthropic_client
        
    async def generate_content_suggestions(
        self,
        current_context: Dict[str, Any],
        max_suggestions: int = 3
    ) -> List[ContentSuggestion]:
        """Generate content suggestions based on memory and context"""
        
        # Get relevant memories
        relevant_content = self._get_relevant_content(current_context)
        
        # Get trending topics from memory
        trending_topics = self._analyze_trending_topics(relevant_content)
        
        # Get personality context
        personality_context = self.personality.get_state_summary()
        
        # Generate suggestions
        suggestions = await self._generate_suggestions(
            relevant_content,
            trending_topics,
            current_context,
            personality_context,
            max_suggestions
        )
        
        return suggestions
        
    def _get_relevant_content(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, List[Episode]]:
        """Get relevant content from memory"""
        relevant = {
            'recent': [],  # Last 24 hours
            'trending': [], # High engagement
            'thematic': [], # Topic-related
            'timeless': []  # High-value evergreen
        }
        
        # Get recent content
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_content = self.memory.get_relevant_episodes(
            type="processed_content",
            limit=20
        )
        
        for content in recent_content:
            # Skip if no timestamp
            if not hasattr(content, 'timestamp'):
                continue
                
            # Categorize content
            if content.timestamp > recent_cutoff:
                relevant['recent'].append(content)
                
            # Check engagement metrics
            if content.content.get('engagement_score', 0) > 0.7:
                relevant['trending'].append(content)
                
            # Check topic alignment
            if self._check_topic_alignment(content, context):
                relevant['thematic'].append(content)
                
            # Check evergreen value
            if self._is_evergreen(content):
                relevant['timeless'].append(content)
                
        return relevant
        
    def _check_topic_alignment(
        self,
        content: Episode,
        context: Dict[str, Any]
    ) -> bool:
        """Check if content aligns with current topics"""
        if 'topics' not in context:
            return False
            
        content_topics = set(
            t['topic'] for t in content.content.get('topics', [])
        )
        context_topics = set(context['topics'])
        
        # Check overlap
        overlap = content_topics & context_topics
        return len(overlap) > 0
        
    def _is_evergreen(self, content: Episode) -> bool:
        """Check if content is evergreen"""
        if 'evergreen_score' in content.content:
            return content.content['evergreen_score'] > 0.7
            
        # Check factors that suggest evergreen content
        indicators = [
            content.content.get('technical_level', 0) > 0.6,
            content.content.get('readability', 0) > 0.7,
            len(content.content.get('key_points', [])) >= 3
        ]
        
        return sum(indicators) >= 2
        
    def _analyze_trending_topics(
        self,
        relevant_content: Dict[str, List[Episode]]
    ) -> List[Dict[str, Any]]:
        """Analyze trending topics from relevant content"""
        topic_stats = defaultdict(lambda: {
            'count': 0,
            'engagement': 0.0,
            'recent_mentions': 0
        })
        
        # Analyze all content
        for category in relevant_content.values():
            for content in category:
                if 'topics' not in content.content:
                    continue
                    
                for topic in content.content['topics']:
                    topic_name = topic['topic']
                    topic_stats[topic_name]['count'] += 1
                    
                    # Track engagement
                    if 'engagement_score' in content.content:
                        topic_stats[topic_name]['engagement'] += \
                            content.content['engagement_score']
                            
                    # Track recency
                    if content in relevant_content['recent']:
                        topic_stats[topic_name]['recent_mentions'] += 1
                        
        # Calculate trend scores
        trending = []
        for topic, stats in topic_stats.items():
            trend_score = (
                stats['count'] * 0.3 +
                stats['engagement'] * 0.4 +
                stats['recent_mentions'] * 0.3
            )
            
            trending.append({
                'topic': topic,
                'trend_score': trend_score,
                'stats': stats
            })
            
        # Sort by trend score
        trending.sort(key=lambda x: x['trend_score'], reverse=True)
        return trending[:10]  # Return top 10 trending topics
        
    async def _generate_suggestions(
        self,
        relevant_content: Dict[str, List[Episode]],
        trending_topics: List[Dict[str, Any]],
        current_context: Dict[str, Any],
        personality_context: Dict[str, Any],
        max_suggestions: int
    ) -> List[ContentSuggestion]:
        """Generate content suggestions using LLM"""
        
        # Prepare content for prompt
        content_summary = {
            'recent_content': [
                {
                    'title': c.content['title'],
                    'summary': c.content['summary'],
                    'key_points': c.content['key_points'],
                    'topics': c.content['topics']
                }
                for c in relevant_content['recent'][:5]
            ],
            'trending_content': [
                {
                    'title': c.content['title'],
                    'summary': c.content['summary'],
                    'engagement_score': c.content.get('engagement_score', 0)
                }
                for c in relevant_content['trending'][:5]
            ],
            'thematic_content': [
                {
                    'title': c.content['title'],
                    'key_points': c.content['key_points'],
                    'topics': c.content['topics']
                }
                for c in relevant_content['thematic'][:5]
            ]
        }
        
        # Create prompt
        prompt = f"""Generate tweet suggestions based on our content knowledge and current context.

Content Knowledge:
{json.dumps(content_summary, indent=2)}

Trending Topics:
{json.dumps(trending_topics, indent=2)}

Current Context:
{json.dumps(current_context, indent=2)}

Personality State:
{json.dumps(personality_context, indent=2)}

Generate {max_suggestions} tweet suggestions that:
1. Leverage our content knowledge
2. Align with trending topics
3. Match our personality
4. Have high engagement potential

Return in JSON format:
{{
    "suggestions": [
        {{
            "tweet_text": "the tweet",
            "source_urls": ["url1", "url2"],
            "topics": ["topic1", "topic2"],
            "confidence": 0.0-1.0,
            "reasoning": "why this tweet works",
            "content_type": "type of content",
            "reference_content": [
                {{
                    "title": "title",
                    "key_points": ["point1", "point2"]
                }}
            ],
            "personality_alignment": 0.0-1.0,
            "timing_score": 0.0-1.0,
            "expected_engagement": 0.0-1.0
        }}
    ],
    "strategy": "overall content strategy"
}}"""
        
        try:
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = json.loads(response.content[0].text)
            
            # Convert to ContentSuggestion objects
            suggestions = []
            for sugg in result['suggestions']:
                suggestions.append(
                    ContentSuggestion(
                        tweet_text=sugg['tweet_text'],
                        source_urls=sugg['source_urls'],
                        topics=sugg['topics'],
                        confidence=sugg['confidence'],
                        reasoning=sugg['reasoning'],
                        content_type=sugg['content_type'],
                        reference_content=sugg['reference_content'],
                        personality_alignment=sugg['personality_alignment'],
                        timing_score=sugg['timing_score'],
                        expected_engagement=sugg['expected_engagement']
                    )
                )
                
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []
            
    async def generate_thread(
        self,
        main_topic: str,
        relevant_content: List[Episode],
        max_tweets: int = 5
    ) -> List[str]:
        """Generate a thread based on content knowledge"""
        
        # Prepare content summary
        content_summary = [
            {
                'title': c.content['title'],
                'summary': c.content['summary'],
                'key_points': c.content['key_points'],
                'topics': c.content['topics']
            }
            for c in relevant_content
        ]
        
        # Create prompt
        prompt = f"""Generate a Twitter thread about {main_topic} using our content knowledge.

Content Knowledge:
{json.dumps(content_summary, indent=2)}

Personality State:
{json.dumps(self.personality.get_state_summary(), indent=2)}

Generate a thread that:
1. Builds a coherent narrative
2. Cites sources appropriately
3. Maintains engagement
4. Fits our personality

Return in JSON format:
{{
    "thread": [
        {{
            "tweet_text": "tweet text",
            "position": 1,
            "key_point": "main point of tweet",
            "source_content": ["title1", "title2"]
        }}
    ],
    "strategy": "thread strategy"
}}"""
        
        try:
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = json.loads(response.content[0].text)
            
            # Extract tweets
            tweets = []
            for tweet in result['thread']:
                tweets.append(tweet['tweet_text'])
                
            return tweets[:max_tweets]
            
        except Exception as e:
            logger.error(f"Failed to generate thread: {e}")
            return []
