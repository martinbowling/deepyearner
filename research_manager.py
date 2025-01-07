from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from content_fetcher import ContentFetcher, ProcessedContent
from collections import defaultdict
import traceback

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
    vibe_signature: Dict[str, float] = None

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
    vibe_alignment: Dict[str, float]

class ResearchManager:
    """Manages research topics and content gathering"""
    
    def __init__(
        self,
        content_fetcher: ContentFetcher,
        anthropic_client: Any,
        memory_system: Any
    ):
        self.content_fetcher = content_fetcher
        self.client = anthropic_client
        self.memory_system = memory_system
        self.research_topics: Dict[str, ResearchTopic] = {}
        self.topic_relationships = defaultdict(set)
        
    async def select_research_topic(
        self,
        timeline_analysis: Dict[str, Any],
        personality_state: Dict[str, Any]
    ) -> Optional[str]:
        """Select a research topic based on timeline and personality"""
        
        try:
            # Create topic selection prompt
            prompt = f"""Based on the following timeline analysis and personality state, suggest a research topic that would be interesting and valuable to explore.

Timeline Analysis:
{json.dumps(timeline_analysis, indent=2)}

Personality State:
{json.dumps(personality_state, indent=2)}

Consider:
1. Current trending topics
2. Knowledge gaps to fill
3. Areas of high engagement
4. Alignment with personality
5. Potential value to followers

Return your suggestion in this exact format:
<topic_suggestion>
{{
    "topic": "the research topic",
    "importance": 0.0-1.0,
    "reasoning": "why this topic now",
    "context": {{
        "related_trends": ["trend1", "trend2"],
        "knowledge_gaps": ["gap1", "gap2"],
        "potential_value": "expected value"
    }},
    "vibe_signature": {{
        "intellectual_depth": 0.0-1.0,
        "chaos_energy": 0.0-1.0,
        "memetic_potential": 0.0-1.0,
        "conversation_quality": 0.0-1.0
    }}
}}
</topic_suggestion>"""

            # Get response from Claude (remove await)
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract and parse response
            content = response.content[0].text.strip()
            if '<topic_suggestion>' in content and '</topic_suggestion>' in content:
                json_str = content.split('<topic_suggestion>')[1].split('</topic_suggestion>')[0].strip()
                suggestion = json.loads(json_str)
                
                # Create new research topic
                topic = ResearchTopic(
                    topic=suggestion['topic'],
                    importance=suggestion['importance'],
                    source='timeline_analysis',
                    context=suggestion['context'],
                    discovery_time=datetime.now(),
                    vibe_signature=suggestion['vibe_signature']
                )
                
                self.research_topics[topic.topic] = topic
                logger.info(f"Selected research topic: {topic.topic} (importance: {topic.importance})")
                return topic.topic
                
            else:
                logger.error("Response missing topic_suggestion XML tags")
                return None
                
        except Exception as e:
            logger.error(f"Failed to select research topic: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def research_topic(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Research a topic and return findings"""
        
        try:
            start_time = datetime.now()
            
            # Get topic object
            topic_obj = self.research_topics.get(topic)
            if not topic_obj:
                topic_obj = ResearchTopic(
                    topic=topic,
                    importance=0.7,
                    source='direct_request',
                    context=context,
                    discovery_time=start_time
                )
                self.research_topics[topic] = topic_obj
            
            # Get relevant content
            content = await self.content_fetcher.get_content(topic)
            
            if not content:
                logger.error(f"No content found for topic: {topic}")
                return None
            
            # Create research prompt
            prompt = f"""Analyze this content and extract key insights about the topic.

Topic: {topic}
Research Depth: {topic_obj.research_depth}
Previous Research Count: {topic_obj.research_count}

Content:
{json.dumps(content, indent=2)}

Context:
{json.dumps(context, indent=2)}

Consider:
1. Main arguments and evidence
2. Emerging patterns and trends
3. Potential implications
4. Areas of consensus/disagreement
5. Knowledge gaps

Return your analysis in this exact format:
<research_insights>
{{
    "key_findings": [
        {{
            "finding": "description",
            "confidence": 0.0-1.0,
            "evidence": ["supporting points"],
            "potential_value": 0.0-1.0
        }}
    ],
    "patterns": [
        {{
            "pattern": "description",
            "strength": 0.0-1.0,
            "examples": ["example1", "example2"]
        }}
    ],
    "new_topics": [
        {{
            "topic": "related topic",
            "relationship": "how it relates",
            "importance": 0.0-1.0
        }}
    ],
    "research_quality": {{
        "depth": 0.0-1.0,
        "breadth": 0.0-1.0,
        "confidence": 0.0-1.0
    }},
    "vibe_signature": {{
        "intellectual_depth": 0.0-1.0,
        "chaos_energy": 0.0-1.0,
        "memetic_potential": 0.0-1.0,
        "conversation_quality": 0.0-1.0
    }}
}}
</research_insights>"""

            # Get response from Claude (remove await)
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract and parse response
            content = response.content[0].text.strip()
            if '<research_insights>' in content and '</research_insights>' in content:
                json_str = content.split('<research_insights>')[1].split('</research_insights>')[0].strip()
                insights = json.loads(json_str)
                
                # Create research session
                session = ResearchSession(
                    topic=topic,
                    start_time=start_time,
                    end_time=datetime.now(),
                    content_found=content,
                    new_topics_discovered=[t['topic'] for t in insights['new_topics']],
                    insights_gained=insights['key_findings'],
                    research_quality=sum(insights['research_quality'].values()) / 3,
                    energy_spent=0.8,
                    vibe_alignment=insights['vibe_signature']
                )
                
                # Update topic state
                self._update_topic_state(topic_obj, session, insights)
                
                # Store research memory
                self._store_research_memory(topic_obj, session, insights)
                
                return insights
                
            else:
                logger.error("Response missing research_insights XML tags")
                return None
                
        except Exception as e:
            logger.error(f"Error researching topic: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _extract_xml_content(self, text: str, tag: str) -> str:
        """Extract content from XML tags"""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        return text[start:end].strip()

    def _update_topic_state(
        self,
        topic: ResearchTopic,
        session: ResearchSession,
        insights: Dict
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
        
        # Update vibe signature
        if not topic.vibe_signature:
            topic.vibe_signature = insights['vibe_signature']
        else:
            for k, v in insights['vibe_signature'].items():
                topic.vibe_signature[k] = (topic.vibe_signature[k] * 0.7) + (v * 0.3)
        
        # Add related topics
        if not topic.related_topics:
            topic.related_topics = []
        topic.related_topics.extend(session.new_topics_discovered)
        
        # Update topic relationships
        for new_topic in insights['new_topics']:
            self.topic_relationships[topic.topic].add(new_topic['topic'])
            self.topic_relationships[new_topic['topic']].add(topic.topic)
        
        # Update topic in storage
        self.research_topics[topic.topic] = topic

    def _store_research_memory(
        self,
        topic: ResearchTopic,
        session: ResearchSession,
        insights: Dict
    ):
        """Store research results in memory"""
        self.memory_system.add_memory({
            'type': 'research',
            'topic': topic.topic,
            'timestamp': session.end_time,
            'content': {
                'findings': insights['key_findings'],
                'patterns': insights['patterns'],
                'research_quality': insights['research_quality'],
                'vibe_signature': insights['vibe_signature']
            },
            'metadata': {
                'research_depth': topic.research_depth,
                'research_count': topic.research_count,
                'session_duration': (session.end_time - session.start_time).seconds,
                'content_count': len(session.content_found)
            }
        })

    def get_related_topics(self, topic: str) -> List[str]:
        """Get topics related to the given topic"""
        return list(self.topic_relationships[topic])

    def get_topic_state(self, topic: str) -> Optional[ResearchTopic]:
        """Get current state of a research topic"""
        return self.research_topics.get(topic) 

    async def get_latest_findings(self) -> Optional[Dict[str, Any]]:
        """Get the latest research findings from memory"""
        try:
            # Get recent research memories
            recent_memories = await self.memory_system.get_recent_memories(
                hours=24,
                limit=10
            )
            
            # Filter for research findings
            research_memories = [
                memory for memory in recent_memories 
                if memory.type == 'research_finding'
            ]
            
            if not research_memories:
                return None
                
            # Get most recent finding
            latest_finding = research_memories[0]
            
            try:
                # Parse the content as JSON
                finding_data = json.loads(latest_finding.content)
                
                # Add metadata
                finding_data.update({
                    'timestamp': latest_finding.timestamp,
                    'context': json.loads(latest_finding.context) if latest_finding.context else {},
                    'source': latest_finding.source
                })
                
                return finding_data
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse research finding content: {latest_finding.content}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest findings: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def add_research_finding(self, finding: Dict[str, Any]) -> None:
        """Add a new research finding to memory"""
        try:
            memory = {
                'type': 'research_finding',
                'content': json.dumps(finding),
                'timestamp': datetime.now().isoformat(),
                'source': finding.get('source', 'research'),
                'context': json.dumps({
                    'topics': finding.get('topics', []),
                    'confidence': finding.get('confidence', 0.0),
                    'insight_type': finding.get('insight_type', 'observation')
                })
            }
            
            await self.memory_system.add_memory(memory)
            
        except Exception as e:
            logger.error(f"Error adding research finding: {str(e)}")
            logger.error(traceback.format_exc()) 