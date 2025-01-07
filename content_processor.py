"""Content processor for analyzing and generating content"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ContentProcessor:
    """Processes content for research and generation"""
    
    def __init__(self, anthropic_client: Any):
        self.client = anthropic_client
        
    async def analyze_content(
        self,
        content: Dict,
        context: Dict[str, Any]
    ) -> Optional[Dict]:
        """Analyze content for insights and patterns"""
        try:
            prompt = f"""Analyze this content to extract insights and patterns.

Content:
{json.dumps(content, indent=2)}

Context:
{json.dumps(context, indent=2)}

Consider:
1. Key insights and findings
2. Patterns and relationships
3. Content quality and credibility
4. Potential applications
5. Vibe and tone analysis

Return analysis in this exact format:
<content_analysis>
{{
    "insights": [
        {{
            "insight": "description",
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
    "content_quality": {{
        "credibility": 0.0-1.0,
        "depth": 0.0-1.0,
        "originality": 0.0-1.0,
        "clarity": 0.0-1.0
    }},
    "applications": [
        {{
            "application": "how to use this",
            "value": 0.0-1.0,
            "requirements": ["req1", "req2"]
        }}
    ],
    "vibe_signature": {{
        "intellectual_depth": 0.0-1.0,
        "chaos_energy": 0.0-1.0,
        "memetic_potential": 0.0-1.0,
        "conversation_quality": 0.0-1.0
    }}
}}
</content_analysis>"""

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return json.loads(self._extract_xml_content(response.content[0].text, "content_analysis"))
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return None
            
    async def generate_content_from_insights(
        self,
        insights: List[Dict],
        context: Dict[str, Any],
        vibe_signature: Dict[str, float]
    ) -> Optional[Dict]:
        """Generate content from research insights"""
        try:
            prompt = f"""Generate engaging content from these research insights.

Insights:
{json.dumps(insights, indent=2)}

Context:
{json.dumps(context, indent=2)}

Vibe Signature:
{json.dumps(vibe_signature, indent=2)}

Generate content that:
1. Synthesizes key insights
2. Matches the vibe signature
3. Provides unique value
4. Encourages engagement
5. Stays within Twitter's limits

Return in this exact format:
<generated_content>
{{
    "content": [
        {{
            "text": "tweet text",
            "type": "insight/commentary/thread",
            "confidence": 0.0-1.0,
            "topics": ["topic1", "topic2"],
            "vibe_alignment": {{
                "intellectual_depth": 0.0-1.0,
                "chaos_energy": 0.0-1.0,
                "memetic_potential": 0.0-1.0,
                "conversation_quality": 0.0-1.0
            }}
        }}
    ],
    "thread_structure": {{
        "main_tweet": 0,
        "supporting_tweets": [1, 2],
        "conclusion_tweet": 3
    }},
    "expected_engagement": {{
        "reply_worthy": 0.0-1.0,
        "retweet_worthy": 0.0-1.0,
        "conversation_starter": 0.0-1.0
    }}
}}
</generated_content>"""

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return json.loads(self._extract_xml_content(response.content[0].text, "generated_content"))
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return None
            
    async def enhance_content(
        self,
        content: str,
        vibe_signature: Dict[str, float],
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Enhance content to better match vibe and context"""
        try:
            prompt = f"""Enhance this content to better match the vibe signature and context.

Original Content:
{content}

Desired Vibe:
{json.dumps(vibe_signature, indent=2)}

Context:
{json.dumps(context, indent=2)}

Enhance the content to:
1. Better match the vibe signature
2. Improve engagement potential
3. Maintain authentic voice
4. Preserve core message
5. Stay within length limits

Return enhanced content in this format:
<enhanced_content>
{{
    "text": "enhanced tweet text",
    "modifications": [
        {{
            "type": "tone/structure/wording",
            "description": "what changed",
            "reason": "why changed"
        }}
    ],
    "vibe_alignment": {{
        "intellectual_depth": 0.0-1.0,
        "chaos_energy": 0.0-1.0,
        "memetic_potential": 0.0-1.0,
        "conversation_quality": 0.0-1.0
    }}
}}
</enhanced_content>"""

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = json.loads(self._extract_xml_content(response.content[0].text, "enhanced_content"))
            return result['text']
            
        except Exception as e:
            logger.error(f"Error enhancing content: {str(e)}")
            return None

    def _extract_xml_content(self, text: str, tag: str) -> str:
        """Extract content from XML tags"""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        return text[start:end].strip()

    async def analyze_engagement_patterns(
        self,
        content: List[Dict],
        engagement_metrics: List[Dict]
    ) -> Optional[Dict]:
        """Analyze patterns in content engagement"""
        try:
            prompt = f"""Analyze engagement patterns in this content.

Content and Metrics:
{json.dumps(list(zip(content, engagement_metrics)), indent=2)}

Analyze:
1. What content performs well
2. Timing patterns
3. Topic performance
4. Engagement types
5. Vibe impact

Return analysis in this format:
<engagement_analysis>
{{
    "patterns": [
        {{
            "pattern": "description",
            "strength": 0.0-1.0,
            "examples": ["content1", "content2"]
        }}
    ],
    "timing_insights": {{
        "best_times": ["time1", "time2"],
        "worst_times": ["time3", "time4"],
        "day_patterns": ["pattern1", "pattern2"]
    }},
    "topic_performance": {{
        "topic": {{
            "engagement_rate": 0.0-1.0,
            "best_format": "format type",
            "audience_resonance": 0.0-1.0
        }}
    }},
    "vibe_impact": {{
        "intellectual_depth": {{
            "impact": -1.0-1.0,
            "context": "when it works/doesn't"
        }},
        "chaos_energy": {{
            "impact": -1.0-1.0,
            "context": "when it works/doesn't"
        }},
        "memetic_potential": {{
            "impact": -1.0-1.0,
            "context": "when it works/doesn't"
        }},
        "conversation_quality": {{
            "impact": -1.0-1.0,
            "context": "when it works/doesn't"
        }}
    }},
    "recommendations": [
        {{
            "recommendation": "what to do",
            "reasoning": "why this works",
            "expected_impact": 0.0-1.0
        }}
    ]
}}
</engagement_analysis>"""

            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return json.loads(self._extract_xml_content(response.content[0].text, "engagement_analysis"))
            
        except Exception as e:
            logger.error(f"Error analyzing engagement: {str(e)}")
            return None
