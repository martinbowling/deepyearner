"""
Content generation system that creates tweets and threads.
Uses unified TwitterClient and MemorySystem interfaces.
"""
import logging
from typing import Dict, List, Optional, Set, Any
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
        self.vibe_signature = {
            'intellectual_depth': 0.0,
            'chaos_energy': 0.0,
            'memetic_potential': 0.0,
            'conversation_quality': 0.0
        }
        self.attention_weights = {
            'topics': {},
            'users': {},
            'conversations': {}
        }

class ContentGenerator:
    """Generates tweet content and threads"""
    
    def __init__(
        self,
        twitter_client: TwitterClient,
        memory_system: MemorySystem,
        prompt_manager: PromptManager,
        anthropic_client: Any
    ):
        self.twitter = twitter_client
        self.memory = memory_system
        self.prompt_manager = prompt_manager
        self.client = anthropic_client
        self.current_patterns = {}
        self.attention_weights = {}

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
            
            # Analyze timeline patterns
            timeline_state = await self._analyze_timeline(current_context.get('timeline', []))
            
            # Generate different types of content based on timeline state
            if timeline_state['energy']['level'] > 0.7:
                # Generate original content
                original = await self._generate_original_content(
                    current_context,
                    recent_memories,
                    timeline_state
                )
                if original:
                    suggestions.append(original)
            
            if timeline_state['vibe_signature']['intellectual_depth'] > 0.6:
                # Generate insight from research
                insight = await self._generate_research_insight(
                    current_context,
                    recent_memories,
                    timeline_state
                )
                if insight:
                    suggestions.append(insight)
            
            # Generate commentary on timeline patterns
            for pattern in timeline_state['patterns']:
                if pattern['strength'] > 0.7:
                    commentary = await self._generate_pattern_commentary(
                        pattern,
                        timeline_state,
                        current_context
                    )
                    if commentary:
                        suggestions.append(commentary)
            
            # Score and rank suggestions
            for suggestion in suggestions:
                self._score_suggestion(suggestion, current_context, timeline_state)
            
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

    async def _analyze_timeline(self, tweets: List[Dict]) -> Dict:
        """Analyze timeline for patterns and context"""
        prompt = f"""Analyze these timeline tweets for patterns, energy, and opportunities.

Tweets: {json.dumps(tweets)}

Return analysis in this exact format:

<timeline_state>
{{
    "patterns": [
        {{
            "type": "content/conversation/engagement",
            "description": "Pattern description",
            "examples": ["tweet1", "tweet2"],
            "strength": 0.8
        }}
    ],
    "energy": {{
        "level": 0.7,
        "type": "intellectual/chaotic/playful/serious",
        "contributing_factors": ["factor1", "factor2"]
    }},
    "attention_weights": {{
        "topics": {{"topic1": 0.8, "topic2": 0.6}},
        "users": {{"user1": 0.9, "user2": 0.7}},
        "conversations": {{"conv1": 0.8, "conv2": 0.5}}
    }},
    "opportunities": [
        {{
            "type": "reply/quote/thread/original",
            "topic": "topic name",
            "context": "why this is an opportunity",
            "urgency": 0.8,
            "potential_impact": 0.7
        }}
    ],
    "vibe_signature": {{
        "intellectual_depth": 0.8,
        "chaos_energy": 0.6,
        "memetic_potential": 0.7,
        "conversation_quality": 0.9
    }}
}}
</timeline_state>"""

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        state = json.loads(self._extract_xml_content(response.content[0].text, "timeline_state"))
        self._update_attention_weights(state)
        return state

    def _update_attention_weights(self, state: Dict):
        """Update attention weights based on new state"""
        # Update topic weights with decay
        for topic, weight in state['attention_weights']['topics'].items():
            current = self.attention_weights.get(topic, 0.0)
            self.attention_weights[topic] = (current * 0.8) + (weight * 0.2)

        # Prune old weights
        self.attention_weights = {
            k: v for k, v in self.attention_weights.items()
            if v > 0.2  # Keep only significant weights
        }

    async def _generate_pattern_commentary(
        self,
        pattern: Dict,
        timeline_state: Dict,
        context: Dict
    ) -> Optional[ContentSuggestion]:
        """Generate commentary on a timeline pattern"""
        prompt = f"""Generate engaging commentary on this timeline pattern.

Pattern: {json.dumps(pattern)}
Timeline State: {json.dumps(timeline_state)}
Context: {json.dumps(context)}

Generate a tweet that:
1. Highlights the pattern insightfully
2. Adds unique perspective
3. Encourages engagement
4. Matches the timeline vibe

Return in XML format:
<tweet_suggestion>
{{
    "text": "tweet text",
    "topics": ["topic1", "topic2"],
    "confidence": 0.0-1.0,
    "vibe_alignment": {{
        "intellectual_depth": 0.0-1.0,
        "chaos_energy": 0.0-1.0,
        "memetic_potential": 0.0-1.0,
        "conversation_quality": 0.0-1.0
    }}
}}
</tweet_suggestion>"""

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        suggestion_data = json.loads(self._extract_xml_content(response.content[0].text, "tweet_suggestion"))
        
        suggestion = ContentSuggestion(
            text=suggestion_data['text'],
            content_type='pattern_commentary',
            confidence=suggestion_data['confidence'],
            topics=set(suggestion_data['topics'])
        )
        suggestion.vibe_signature = suggestion_data['vibe_alignment']
        
        return suggestion

    def _extract_xml_content(self, text: str, tag: str) -> str:
        """Extract content from XML tags"""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        return text[start:end].strip()

    def _score_suggestion(
        self,
        suggestion: ContentSuggestion,
        context: Dict,
        timeline_state: Dict
    ):
        """Score a content suggestion with enhanced metrics"""
        try:
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Calculate vibe alignment
            vibe_alignment = sum(
                abs(suggestion.vibe_signature[k] - timeline_state['vibe_signature'][k])
                for k in suggestion.vibe_signature
            ) / len(suggestion.vibe_signature)
            
            # Calculate topic alignment
            topic_alignment = sum(
                self.attention_weights.get(topic, 0)
                for topic in suggestion.topics
            ) / max(len(suggestion.topics), 1)
            
            # Calculate personality alignment
            suggestion.personality_alignment = min(
                1.0,
                personality_state['energy'] * 0.3 +
                vibe_alignment * 0.4 +
                topic_alignment * 0.3
            )
            
            # Calculate timing score
            current_hour = datetime.now().hour
            suggestion.timing_score = 0.7  # Base score
            if 9 <= current_hour <= 17:  # Business hours
                suggestion.timing_score += 0.2
            if suggestion.content_type == 'pattern_commentary':
                suggestion.timing_score += 0.1
            
            # Calculate expected engagement
            suggestion.expected_engagement = min(
                1.0,
                suggestion.confidence * 0.3 +
                suggestion.personality_alignment * 0.4 +
                suggestion.timing_score * 0.3
            )
            
        except Exception as e:
            logger.error(f"Error scoring suggestion: {str(e)}")

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
            
            # Convert any sets in personality state to lists for serialization
            serialized_state = self._serialize_for_json(personality_state)
            
            # Generate thread outline
            outline = await self._generate_thread_outline(
                main_topic,
                relevant_content,
                serialized_state
            )
            
            # Generate individual tweets
            tweets = []
            for point in outline[:max_tweets]:
                tweet = await self._generate_tweet_from_point(
                    point,
                    main_topic,
                    serialized_state
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
                        'personality_state': serialized_state,
                        'relevant_content': relevant_content
                    }
                ))
            
            return tweets
            
        except Exception as e:
            logger.error(f"Error generating thread: {str(e)}")
            return []
    
    def _serialize_for_json(self, data: Any) -> Any:
        """Recursively serialize data for JSON, handling sets"""
        if isinstance(data, dict):
            return {k: self._serialize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_json(item) for item in data]
        elif isinstance(data, set):
            return list(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
    
    async def _generate_original_content(
        self,
        context: Dict,
        recent_memories: List[Memory],
        timeline_state: Dict
    ) -> Optional[ContentSuggestion]:
        """Generate original content based on context"""
        try:
            prompt = f"""Generate original content based on the current context and timeline state.

Timeline State:
{json.dumps(timeline_state)}

Context:
{json.dumps(context)}

Recent Memory Topics:
{json.dumps([list(m.content.get('topics', [])) for m in recent_memories])}

Generate content that:
1. Aligns with the current timeline vibe
2. Builds on recent memory topics
3. Adds unique perspective
4. Encourages engagement

Return in this exact format:
<original_content>
{{
    "text": "tweet text",
    "topics": ["topic1", "topic2"],
    "confidence": 0.0-1.0,
    "content_type": "observation/insight/question",
    "vibe_alignment": {{
        "intellectual_depth": 0.0-1.0,
        "chaos_energy": 0.0-1.0,
        "memetic_potential": 0.0-1.0,
        "conversation_quality": 0.0-1.0
    }},
    "expected_impact": {{
        "novelty": 0.0-1.0,
        "resonance": 0.0-1.0,
        "timeliness": 0.0-1.0
    }}
}}
</original_content>"""

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            content_data = json.loads(self._extract_xml_content(response.content[0].text, "original_content"))
            
            suggestion = ContentSuggestion(
                text=content_data['text'],
                content_type=content_data['content_type'],
                confidence=content_data['confidence'],
                topics=set(content_data['topics'])
            )
            suggestion.vibe_signature = content_data['vibe_alignment']
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error generating original content: {str(e)}")
            return None
    
    async def _generate_research_insight(
        self,
        context: Dict,
        recent_memories: List[Memory],
        timeline_state: Dict
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
            
            # Get research insights
            insights = []
            for memory in research_memories[:3]:  # Use last 3 research memories
                if 'findings' in memory.content:
                    insights.extend(memory.content['findings'])
            
            if not insights:
                return None
                
            prompt = f"""Generate an insightful tweet from our research findings.

Research Insights:
{json.dumps(insights)}

Timeline State:
{json.dumps(timeline_state)}

Context:
{json.dumps(context)}

Generate content that:
1. Synthesizes research insights
2. Matches timeline vibe
3. Provides unique value
4. Encourages intellectual discourse
5. Stays within Twitter limits

Return in this exact format:
<research_content>
{{
    "text": "tweet text",
    "topics": ["topic1", "topic2"],
    "confidence": 0.0-1.0,
    "referenced_findings": ["finding1", "finding2"],
    "vibe_alignment": {{
        "intellectual_depth": 0.0-1.0,
        "chaos_energy": 0.0-1.0,
        "memetic_potential": 0.0-1.0,
        "conversation_quality": 0.0-1.0
    }},
    "value_add": {{
        "insight_quality": 0.0-1.0,
        "practical_value": 0.0-1.0,
        "conversation_potential": 0.0-1.0
    }}
}}
</research_content>"""

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            content_data = json.loads(self._extract_xml_content(response.content[0].text, "research_content"))
            
            suggestion = ContentSuggestion(
                text=content_data['text'],
                content_type='research_insight',
                confidence=content_data['confidence'],
                topics=set(content_data['topics']),
                reference_content={
                    'findings': content_data['referenced_findings'],
                    'value_metrics': content_data['value_add']
                }
            )
            suggestion.vibe_signature = content_data['vibe_alignment']
            
            return suggestion
            
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
    
    async def _generate_thread_outline(
        self,
        topic: str,
        content: List[Dict],
        personality_state: Dict
    ) -> List[Dict]:
        """Generate outline for a thread"""
        try:
            prompt = f"""Generate an engaging thread outline about this topic.

Topic: {topic}
Content: {json.dumps(content)}
Personality: {json.dumps(personality_state)}

Create a thread that:
1. Builds narrative flow
2. Maintains engagement
3. Provides value
4. Encourages interaction
5. Fits Twitter format

Return in this exact format:
<thread_outline>
{{
    "title_tweet": {{
        "text": "hook tweet text",
        "purpose": "why this hook works"
    }},
    "main_points": [
        {{
            "text": "tweet text",
            "key_point": "main idea",
            "supporting_content": ["content1", "content2"],
            "engagement_hook": "why engaging"
        }}
    ],
    "conclusion_tweet": {{
        "text": "final tweet text",
        "call_to_action": "what readers should do",
        "conversation_starter": "question/prompt for engagement"
    }},
    "thread_metrics": {{
        "intellectual_depth": 0.0-1.0,
        "narrative_flow": 0.0-1.0,
        "engagement_potential": 0.0-1.0
    }}
}}
</thread_outline>"""

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            outline = json.loads(self._extract_xml_content(response.content[0].text, "thread_outline"))
            
            # Convert to list format
            tweets = [
                {
                    'point': 'Title',
                    'content': outline['title_tweet']['text'],
                    'purpose': outline['title_tweet']['purpose']
                }
            ]
            
            for point in outline['main_points']:
                tweets.append({
                    'point': point['key_point'],
                    'content': point['text'],
                    'supporting_content': point['supporting_content'],
                    'engagement_hook': point['engagement_hook']
                })
                
            tweets.append({
                'point': 'Conclusion',
                'content': outline['conclusion_tweet']['text'],
                'call_to_action': outline['conclusion_tweet']['call_to_action']
            })
            
            return tweets
            
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
            prompt = f"""Generate an engaging tweet for this thread point.

Point: {json.dumps(point)}
Topic: {topic}
Personality: {json.dumps(personality_state)}

Generate a tweet that:
1. Conveys the key point clearly
2. Maintains thread narrative
3. Encourages reading next tweet
4. Matches personality voice
5. Fits Twitter format

Return in this exact format:
<tweet_text>
{{
    "text": "the tweet text",
    "metrics": {{
        "clarity": 0.0-1.0,
        "engagement": 0.0-1.0,
        "personality_match": 0.0-1.0
    }}
}}
</tweet_text>"""

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            tweet_data = json.loads(self._extract_xml_content(response.content[0].text, "tweet_text"))
            return tweet_data['text']
            
        except Exception as e:
            logger.error(f"Error generating tweet from point: {str(e)}")
            return None

    async def post_thread(
        self,
        main_topic: str,
        relevant_content: List[Dict],
        max_tweets: int = 5
    ) -> List[Dict]:
        """Generate and post a thread about a topic"""
        try:
            # Generate thread content
            tweets = await self.generate_thread(main_topic, relevant_content, max_tweets)
            
            if not tweets:
                logger.error("Failed to generate thread content")
                return []
            
            # Post thread using twitter_utils
            thread_tweets = await self.twitter.post_thread(tweets)
            
            if thread_tweets:
                # Store thread in memory
                self.memory.add_memory(Memory(
                    id=f"posted_thread_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    type='posted_thread',
                    content={
                        'topic': main_topic,
                        'tweets': thread_tweets,
                        'tweet_ids': [t.get('data', {}).get('id') for t in thread_tweets]
                    },
                    context={
                        'relevant_content': relevant_content
                    }
                ))
            
            return thread_tweets
            
        except Exception as e:
            logger.error(f"Error posting thread: {str(e)}")
            return []
