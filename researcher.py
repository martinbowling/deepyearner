from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import random
import asyncio
from collections import defaultdict
from memory_system import MemorySystem, Episode
from content_processor import ContentProcessor
from content_fetcher import ContentFetcher
from search_system import SearchSystem, SearchConfig
from personality_system import PersonalitySystem

logger = logging.getLogger(__name__)

@dataclass
class ResearchTopic:
    """Topic to research"""
    topic: str
    importance: float
    source: str  # where we found this topic
    context: Dict[str, Any]
    discovery_time: datetime
    last_researched: Optional[datetime] = None
    research_count: int = 0
    related_topics: List[str] = None
    research_depth: float = 0.0  # 0-1, how deeply we've researched

@dataclass
class ResearchSession:
    """Results from a research session"""
    topic: str
    start_time: datetime
    end_time: datetime
    content_found: List[Dict]
    new_topics_discovered: List[str]
    insights_gained: List[Dict]
    research_quality: float
    energy_spent: float

class ResearchManager:
    """Manages proactive research and learning"""
    
    def __init__(
        self,
        memory_system: MemorySystem,
        content_fetcher: ContentFetcher,
        personality: PersonalitySystem,
        anthropic_client: Any,
        search_config: SearchConfig
    ):
        self.memory = memory_system
        self.fetcher = content_fetcher
        self.personality = personality
        self.client = anthropic_client
        self.search_config = search_config
        self.processor = ContentProcessor(anthropic_client)
        
        # Research state
        self.research_topics: Dict[str, ResearchTopic] = {}
        self.daily_research_count = 0
        self.last_research_reset = datetime.now()
        
    async def manage_research(self) -> Optional[ResearchSession]:
        """Main research management loop"""
        
        # Reset daily counter if needed
        self._check_daily_reset()
        
        # Check if we should research now
        if not self._should_research():
            return None
            
        # Get research topic
        topic = await self._select_research_topic()
        if not topic:
            return None
            
        # Conduct research
        session = await self._conduct_research(topic)
        
        # Update state
        if session:
            self.daily_research_count += 1
            self._update_topic_state(topic, session)
            
        return session
        
    def _check_daily_reset(self):
        """Reset daily counters if needed"""
        now = datetime.now()
        if now.date() > self.last_research_reset.date():
            self.daily_research_count = 0
            self.last_research_reset = now
            
    def _should_research(self) -> bool:
        """Determine if we should research now"""
        # Check daily limit
        if self.daily_research_count >= 15:  # Allow up to 15 research sessions
            return False
            
        # Check energy
        if self.personality.energy.mental_energy < 0.3:
            return False
            
        # Random chance based on personality
        personality_state = self.personality.get_state_summary()
        research_drive = personality_state.get('research_drive', 0.5)
        
        return random.random() < research_drive
        
    async def _select_research_topic(self) -> Optional[ResearchTopic]:
        """Select next topic to research"""
        # Get candidate topics
        candidates = await self._get_research_candidates()
        
        if not candidates:
            # Generate some random topics if we have none
            await self._generate_random_topics()
            candidates = await self._get_research_candidates()
            
        if not candidates:
            return None
            
        # Score candidates
        scored_topics = []
        for topic in candidates:
            score = self._score_topic(topic)
            scored_topics.append((score, topic))
            
        # Select topic
        if scored_topics:
            scored_topics.sort(reverse=True)
            return scored_topics[0][1]
            
        return None
        
    async def _get_research_candidates(self) -> List[ResearchTopic]:
        """Get candidate topics for research"""
        candidates = []
        
        # Get topics from memory
        memory_topics = self._get_memory_topics()
        candidates.extend(memory_topics)
        
        # Get topics from current research list
        for topic in self.research_topics.values():
            if self._is_topic_ready(topic):
                candidates.append(topic)
                
        return candidates
        
    def _get_memory_topics(self) -> List[ResearchTopic]:
        """Get potential topics from memory"""
        topics = []
        
        # Get recent content
        recent_content = self.memory.get_relevant_episodes(
            type="processed_content",
            limit=50
        )
        
        # Extract topics
        topic_mentions = defaultdict(list)
        for content in recent_content:
            if 'topics' not in content.content:
                continue
                
            for topic_data in content.content['topics']:
                topic = topic_data['topic']
                topic_mentions[topic].append(content)
                
        # Convert to research topics
        for topic, mentions in topic_mentions.items():
            if topic not in self.research_topics:
                importance = len(mentions) * 0.1
                topics.append(ResearchTopic(
                    topic=topic,
                    importance=importance,
                    source='memory',
                    context={
                        'mentions': len(mentions),
                        'sources': [m.content['url'] for m in mentions[:3]]
                    },
                    discovery_time=datetime.now()
                ))
                
        return topics
        
    async def _generate_random_topics(self):
        """Generate random topics to research"""
        try:
            # Get personality context
            personality_state = self.personality.get_state_summary()
            
            # Create prompt
            prompt = f"""Generate interesting topics to research based on our personality and interests.

Personality State:
{json.dumps(personality_state, indent=2)}

Generate diverse topics that:
1. Match our interests
2. Could lead to valuable insights
3. Have depth for exploration
4. Could engage our audience

Return in JSON format:
{{
    "topics": [
        {{
            "topic": "topic name",
            "importance": 0.0-1.0,
            "reasoning": "why interesting",
            "potential_angles": ["angle1", "angle2"]
        }}
    ]
}}"""
            
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = json.loads(response.content[0].text)
            
            # Add topics
            for topic_data in result['topics']:
                if topic_data['topic'] not in self.research_topics:
                    self.research_topics[topic_data['topic']] = ResearchTopic(
                        topic=topic_data['topic'],
                        importance=topic_data['importance'],
                        source='generated',
                        context={
                            'reasoning': topic_data['reasoning'],
                            'angles': topic_data['potential_angles']
                        },
                        discovery_time=datetime.now()
                    )
                    
        except Exception as e:
            logger.error(f"Failed to generate random topics: {e}")
            
    def _score_topic(self, topic: ResearchTopic) -> float:
        """Score topic for research priority"""
        score = topic.importance
        
        # Reduce score for recently researched topics
        if topic.last_researched:
            hours_since = (
                datetime.now() - topic.last_researched
            ).total_seconds() / 3600
            recency_penalty = min(1.0, hours_since / 24)
            score *= recency_penalty
            
        # Reduce score for deeply researched topics
        depth_penalty = 1.0 - (topic.research_depth * 0.7)
        score *= depth_penalty
        
        # Boost score for trending topics
        if self._is_topic_trending(topic.topic):
            score *= 1.5
            
        return score
        
    def _is_topic_trending(self, topic: str) -> bool:
        """Check if topic is trending"""
        recent_content = self.memory.get_relevant_episodes(
            type="processed_content",
            limit=20
        )
        
        mentions = sum(
            1 for c in recent_content
            if any(
                t['topic'] == topic
                for t in c.content.get('topics', [])
            )
        )
        
        return mentions >= 3
        
    def _is_topic_ready(self, topic: ResearchTopic) -> bool:
        """Check if topic is ready for research"""
        if not topic.last_researched:
            return True
            
        hours_since = (
            datetime.now() - topic.last_researched
        ).total_seconds() / 3600
        
        # More researched topics need more time between sessions
        required_hours = 4 + (topic.research_depth * 20)
        
        return hours_since >= required_hours
        
    async def _conduct_research(
        self,
        topic: ResearchTopic
    ) -> Optional[ResearchSession]:
        """Conduct research on a topic"""
        try:
            start_time = datetime.now()
            
            # Initialize search
            async with SearchSystem(self.search_config) as search:
                # Search for content
                results = await search.search_with_context(
                    query=topic.topic,
                    context=topic.context,
                    max_results=10
                )
                
                # Process results
                processed_content = await self.fetcher.process_search_results(
                    results,
                    {
                        'topic': topic.topic,
                        'context': topic.context,
                        'research_depth': topic.research_depth
                    }
                )
                
            # Extract insights
            insights = await self._extract_insights(
                topic,
                processed_content
            )
            
            # Calculate research quality
            quality = self._calculate_research_quality(
                processed_content,
                insights
            )
            
            # Update energy
            energy_spent = 0.2 + (quality * 0.3)
            self.personality.energy.update_energy(
                'mental',
                energy_spent
            )
            
            # Create session record
            session = ResearchSession(
                topic=topic.topic,
                start_time=start_time,
                end_time=datetime.now(),
                content_found=[
                    {
                        'url': c.url,
                        'title': c.title,
                        'summary': c.summary
                    }
                    for c in processed_content
                ],
                new_topics_discovered=self._extract_new_topics(
                    processed_content
                ),
                insights_gained=insights,
                research_quality=quality,
                energy_spent=energy_spent
            )
            
            # Store research session in memory
            self.memory.add_episode(
                type="research_session",
                content={
                    'topic': topic.topic,
                    'duration_minutes': (
                        session.end_time - session.start_time
                    ).total_seconds() / 60,
                    'content_found': session.content_found,
                    'insights': session.insights_gained,
                    'quality': session.research_quality,
                    'energy_spent': session.energy_spent
                },
                context={
                    'personality_state': self.personality.get_state_summary(),
                    'research_depth': topic.research_depth
                },
                participants=set()
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Research session failed: {e}")
            return None
            
    async def _extract_insights(
        self,
        topic: ResearchTopic,
        content: List[Any]
    ) -> List[Dict]:
        """Extract insights from research"""
        try:
            # Prepare content summary
            content_summary = [
                {
                    'title': c.title,
                    'summary': c.summary,
                    'key_points': c.key_points,
                    'topics': c.topics
                }
                for c in content
            ]
            
            # Create prompt
            prompt = f"""Analyze this research content and extract key insights.

Topic: {topic.topic}
Research Depth: {topic.research_depth}

Content Found:
{json.dumps(content_summary, indent=2)}

Extract:
1. Key insights and findings
2. Patterns and connections
3. Questions for further research
4. Potential applications

Return in JSON format:
{{
    "insights": [
        {{
            "insight": "description",
            "confidence": 0.0-1.0,
            "supporting_content": ["title1", "title2"],
            "potential_value": "how this could be valuable"
        }}
    ],
    "patterns": [
        {{
            "pattern": "description",
            "evidence": ["evidence1", "evidence2"]
        }}
    ],
    "questions": ["question1", "question2"],
    "applications": ["application1", "application2"]
}}"""
            
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1500,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return json.loads(response.content[0].text)
            
        except Exception as e:
            logger.error(f"Failed to extract insights: {e}")
            return []
            
    def _calculate_research_quality(
        self,
        content: List[Any],
        insights: List[Dict]
    ) -> float:
        """Calculate quality of research session"""
        if not content:
            return 0.0
            
        factors = [
            len(content) * 0.1,  # Amount of content
            len(insights) * 0.2,  # Number of insights
            
            # Average content quality
            sum(c.readability_score for c in content) / len(content) * 0.3,
            
            # Insight quality
            sum(
                i.get('confidence', 0)
                for i in insights
            ) / max(len(insights), 1) * 0.4
        ]
        
        return min(1.0, sum(factors))
        
    def _extract_new_topics(self, content: List[Any]) -> List[str]:
        """Extract new topics from research"""
        topics = set()
        for c in content:
            for topic in c.topics:
                if topic['topic'] not in self.research_topics:
                    topics.add(topic['topic'])
        return list(topics)
        
    def _update_topic_state(
        self,
        topic: ResearchTopic,
        session: ResearchSession
    ):
        """Update topic state after research"""
        topic.last_researched = session.end_time
        topic.research_count += 1
        
        # Update research depth
        depth_increase = session.research_quality * 0.2
        topic.research_depth = min(
            1.0,
            topic.research_depth + depth_increase
        )
        
        # Add related topics
        if not topic.related_topics:
            topic.related_topics = []
        topic.related_topics.extend(session.new_topics_discovered)
        
        # Update topic in storage
        self.research_topics[topic.topic] = topic
