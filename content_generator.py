"""
Content generation system that creates tweets and threads.
Uses unified TwitterClient and MemorySystem interfaces.
"""
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import json
import random

from twitter_utils import TwitterClient
from memory_system import MemorySystem, Memory
from prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class ContentSuggestion:
    """Represents a content suggestion for posting"""
    def __init__(
        self,
        text: str,
        content_type: str,
        confidence: float,
        topics: Set[str],
        source_urls: Optional[List[str]] = None,
        reference_content: Optional[Dict] = None
    ):
        self.text = text
        self.content_type = content_type
        self.confidence = confidence
        self.topics = topics
        self.source_urls = source_urls or []
        self.reference_content = reference_content or {}
        self.personality_alignment = 0.0
        self.timing_score = 0.0
        self.expected_engagement = 0.0

class ContentGenerator:
    """Generates tweet content and threads"""
    
    def __init__(
        self,
        twitter_client: TwitterClient,
        memory_system: MemorySystem,
        prompt_manager: PromptManager
    ):
        self.twitter = twitter_client
        self.memory = memory_system
        self.prompt_manager = prompt_manager
    
    async def generate_content_suggestions(
        self,
        current_context: Dict,
        max_suggestions: int = 3
    ) -> List[ContentSuggestion]:
        """Generate content suggestions based on context"""
        try:
            suggestions = []
            
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Get relevant memories
            recent_memories = self.memory.get_recent_memories(limit=20)
            
            # Generate different types of content
            if personality_state['energy'] > 0.7:
                # Generate original content
                original = await self._generate_original_content(
                    current_context,
                    recent_memories
                )
                if original:
                    suggestions.append(original)
            
            if personality_state['focus'] > 0.6:
                # Generate insight from research
                insight = await self._generate_research_insight(
                    current_context,
                    recent_memories
                )
                if insight:
                    suggestions.append(insight)
            
            if len(suggestions) < max_suggestions:
                # Generate commentary on timeline
                commentary = await self._generate_commentary(
                    current_context,
                    recent_memories
                )
                if commentary:
                    suggestions.append(commentary)
            
            # Score and rank suggestions
            for suggestion in suggestions:
                self._score_suggestion(suggestion, current_context)
            
            suggestions.sort(
                key=lambda x: (
                    x.confidence * 0.3 +
                    x.personality_alignment * 0.3 +
                    x.expected_engagement * 0.4
                ),
                reverse=True
            )
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error generating content suggestions: {str(e)}")
            return []
    
    async def generate_thread(
        self,
        main_topic: str,
        relevant_content: List[Dict],
        max_tweets: int = 5
    ) -> List[str]:
        """Generate a thread about a topic"""
        try:
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Generate thread outline
            outline = await self._generate_thread_outline(
                main_topic,
                relevant_content,
                personality_state
            )
            
            # Generate individual tweets
            tweets = []
            for point in outline[:max_tweets]:
                tweet = await self._generate_tweet_from_point(
                    point,
                    main_topic,
                    personality_state
                )
                if tweet:
                    tweets.append(tweet)
            
            # Store thread plan in memory
            if tweets:
                self.memory.add_memory(Memory(
                    id=f"thread_plan_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    type='thread_plan',
                    content={
                        'topic': main_topic,
                        'tweets': tweets,
                        'outline': outline
                    },
                    context={
                        'personality_state': personality_state,
                        'relevant_content': relevant_content
                    }
                ))
            
            return tweets
            
        except Exception as e:
            logger.error(f"Error generating thread: {str(e)}")
            return []
    
    async def _generate_original_content(
        self,
        context: Dict,
        recent_memories: List[Memory]
    ) -> Optional[ContentSuggestion]:
        """Generate original content based on context"""
        try:
            # Get relevant topics from memory
            topics = set()
            for memory in recent_memories:
                if 'topics' in memory.content:
                    topics.update(memory.content['topics'])
            
            # Generate content
            # This would typically use an LLM
            text = f"Here's an interesting thought about {random.choice(list(topics))}..."
            
            return ContentSuggestion(
                text=text,
                content_type='original',
                confidence=0.8,
                topics=topics
            )
            
        except Exception as e:
            logger.error(f"Error generating original content: {str(e)}")
            return None
    
    async def _generate_research_insight(
        self,
        context: Dict,
        recent_memories: List[Memory]
    ) -> Optional[ContentSuggestion]:
        """Generate insight from research"""
        try:
            # Find research memories
            research_memories = [
                m for m in recent_memories
                if m.type == 'research'
            ]
            
            if not research_memories:
                return None
            
            # Generate insight
            # This would typically use an LLM
            memory = random.choice(research_memories)
            text = f"Based on recent research: {str(memory.content)[:100]}..."
            
            return ContentSuggestion(
                text=text,
                content_type='insight',
                confidence=0.7,
                topics=set(memory.content.get('topics', [])),
                reference_content=memory.content
            )
            
        except Exception as e:
            logger.error(f"Error generating research insight: {str(e)}")
            return None
    
    async def _generate_commentary(
        self,
        context: Dict,
        recent_memories: List[Memory]
    ) -> Optional[ContentSuggestion]:
        """Generate commentary on timeline content"""
        try:
            # Get timeline analysis
            timeline = context.get('timeline_analysis', {})
            
            if not timeline:
                return None
            
            # Generate commentary
            # This would typically use an LLM
            text = f"Interesting trend in the timeline: {random.choice(timeline.get('trends', ['something']))}"
            
            return ContentSuggestion(
                text=text,
                content_type='commentary',
                confidence=0.6,
                topics=set(timeline.get('topics', []))
            )
            
        except Exception as e:
            logger.error(f"Error generating commentary: {str(e)}")
            return None
    
    def _score_suggestion(self, suggestion: ContentSuggestion, context: Dict):
        """Score a content suggestion"""
        try:
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Calculate personality alignment
            suggestion.personality_alignment = min(
                1.0,
                personality_state['energy'] * 0.5 +
                personality_state['focus'] * 0.3 +
                (1.0 if suggestion.content_type == 'insight' else 0.7) * 0.2
            )
            
            # Calculate timing score
            current_hour = datetime.now().hour
            suggestion.timing_score = 0.7  # Base score
            if 9 <= current_hour <= 17:  # Business hours
                suggestion.timing_score += 0.2
            if suggestion.content_type == 'commentary':
                suggestion.timing_score += 0.1
            
            # Calculate expected engagement
            suggestion.expected_engagement = min(
                1.0,
                suggestion.confidence * 0.4 +
                suggestion.personality_alignment * 0.3 +
                suggestion.timing_score * 0.3
            )
            
        except Exception as e:
            logger.error(f"Error scoring suggestion: {str(e)}")
    
    async def _generate_thread_outline(
        self,
        topic: str,
        content: List[Dict],
        personality_state: Dict
    ) -> List[Dict]:
        """Generate outline for a thread"""
        try:
            # This would typically use an LLM
            return [
                {'point': 'Introduction', 'content': f"Let's talk about {topic}"},
                {'point': 'Main Point 1', 'content': 'First important point...'},
                {'point': 'Main Point 2', 'content': 'Second important point...'},
                {'point': 'Conclusion', 'content': 'In conclusion...'}
            ]
        except Exception as e:
            logger.error(f"Error generating thread outline: {str(e)}")
            return []
    
    async def _generate_tweet_from_point(
        self,
        point: Dict,
        topic: str,
        personality_state: Dict
    ) -> Optional[str]:
        """Generate tweet text from an outline point"""
        try:
            # This would typically use an LLM
            return f"{point['point']}: {point['content']}"
        except Exception as e:
            logger.error(f"Error generating tweet from point: {str(e)}")
            return None
