#!/usr/bin/env python3

import os
import sys
import logging
import asyncio
import traceback
import json
import argparse
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime
import anthropic
from twitter_utils import TwitterClient
from memory_system import MemorySystem, Memory
from personality_system import PersonalitySystem, PersonalityEvent
from research_manager import ResearchManager
from content_fetcher import ContentFetcher
from content_guidelines import get_guidelines_prompt
from config import ANTHROPIC_API_KEY, VECTOR_DB_PATH
import uuid
import random

logger = logging.getLogger(__name__)

def init_db():
    """Initialize database tables"""
    try:
        conn = sqlite3.connect("bot.db")
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tweet_engagement (
                tweet_id TEXT,
                user_id TEXT,
                engagement_type TEXT,
                timestamp TEXT,
                PRIMARY KEY (tweet_id, user_id, engagement_type)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                user_id TEXT,
                interaction_type TEXT,
                timestamp TEXT,
                context TEXT,
                PRIMARY KEY (user_id, interaction_type, timestamp)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                timestamp TEXT,
                context TEXT,
                participants TEXT,
                embedding TEXT,
                source TEXT
            )
        """)
        
        conn.commit()
        return conn
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def migrate_db(conn):
    """Run any necessary database migrations"""
    try:
        cursor = conn.cursor()
        
        # Check if source column exists
        cursor.execute("PRAGMA table_info(memories)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'source' not in columns:
            logger.info("Adding source column to memories table")
            cursor.execute("ALTER TABLE memories ADD COLUMN source TEXT")
            
        conn.commit()
    except Exception as e:
        logger.error(f"Error in database migration: {e}")
        raise

async def single_iteration(
    twitter_client: TwitterClient,
    personality_system: PersonalitySystem,
    anthropic_client: Any,
    research_manager: ResearchManager
) -> Optional[int]:  # Return the suggested pause time
    """Run a single iteration of the bot's main loop"""
    try:
        # Get more timeline data for better analysis
        timeline_data = twitter_client.get_home_timeline(count=50)
        if not timeline_data:
            logger.warning("No timeline data received")
            return
            
        # Log timeline stats
        logger.info(f"Retrieved {len(timeline_data)} tweets from timeline")
        
        # Analyze timeline
        timeline_analysis = await analyze_timeline(timeline_data, anthropic_client)
        if not timeline_analysis:
            logger.warning("No timeline analysis generated")
            return
            
        # Create timeline event with proper structure
        timeline_event = {
            'type': 'timeline_analysis',
            'data': timeline_analysis,
            'timestamp': datetime.now().isoformat(),
            'tweet_count': len(timeline_data),
            'source': 'timeline',
            'context': {
                'analyzed_tweets': len(timeline_data),
                'energy_level': timeline_analysis.get('energy_level', 0.5),
                'mood': timeline_analysis.get('mood', 'neutral')
            }
        }
        
        # Add event to personality system
        await personality_system.add_event(timeline_event)
        
        # Get current state
        state = await personality_system.get_state()
        
        # Decide whether to tweet
        should_tweet = await should_tweet_now(state, timeline_analysis, anthropic_client)
        if should_tweet:
            logger.info("Deciding to tweet based on current state")
            content_data = await generate_and_post_tweet(
                twitter_client, 
                state, 
                timeline_analysis, 
                research_manager,
                anthropic_client
            )
            
            # Return the suggested pause time if we tweeted
            if content_data and 'suggested_pause' in content_data:
                logger.info(f"Next iteration in {content_data['suggested_pause']} seconds due to: {content_data['pause_reasoning']}")
                return content_data['suggested_pause']
                
        return 300  # Default 5 minute pause if we didn't tweet
            
    except Exception as e:
        logger.error(f"Error in iteration: {str(e)}")
        logger.error(traceback.format_exc())
        return 300  # Default pause on error

async def run(continuous: bool = True) -> None:
    """Run the bot"""
    db_conn = None
    try:
        # Initialize components
        twitter_client = TwitterClient()
        
        # Initialize database with sqlite3
        db_conn = init_db()
        
        # Initialize Anthropic client with proper API key format
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = ANTHROPIC_API_KEY  # Try config file if env var not set
            
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or config")
            
        if not api_key.startswith("sk-ant"):
            raise ValueError("Invalid Anthropic API key format. Must start with 'sk-ant'")
            
        client = anthropic.Client(api_key=api_key)
        
        # Initialize memory and personality systems
        memory_system = MemorySystem(
            db=db_conn,
            persist_directory=str(VECTOR_DB_PATH)
        )
        personality_system = PersonalitySystem(memory_system, client)
        
        # Initialize research components
        content_fetcher = ContentFetcher()
        research_manager = ResearchManager(
            content_fetcher=content_fetcher,
            anthropic_client=client,
            memory_system=memory_system
        )
        
        while True:
            try:
                # Run iteration and get suggested pause time
                pause_duration = await single_iteration(
                    twitter_client,
                    personality_system,
                    client,
                    research_manager
                )
                
                if not continuous:
                    break
                    
                # Use the suggested pause duration
                logger.info(f"Sleeping for {pause_duration} seconds")
                await asyncio.sleep(pause_duration)
                
            except Exception as e:
                logger.error(f"Error in iteration: {e}")
                if not continuous:
                    raise
                await asyncio.sleep(60)  # Wait 1 minute on error
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        if db_conn:
            db_conn.close()

async def analyze_timeline(timeline: List[Dict[str, Any]], anthropic_client: Any) -> Dict[str, Any]:
    """Analyze the timeline using Claude"""
    try:
        # Format timeline data for better readability
        timeline_text = "Timeline Data:\n"
        for tweet in timeline:
            timeline_text += f"""
Tweet by @{tweet['user']['screen_name']}:
Text: {tweet['text']}
Created at: {tweet['created_at']}
Metrics: {json.dumps(tweet['metrics'], indent=2)}
-------------------
"""

        # Enhanced analysis prompt
        prompt = f"""Analyze these tweets from our timeline to guide content creation and engagement.

{timeline_text}

Analyze for:
1. Content Patterns
   - Popular formats (threads, images, polls)
   - Successful engagement tactics
   - Time of day patterns

2. Conversation Context
   - Active discussions
   - Recent news/events
   - Emerging debates

3. Community Dynamics
   - Key influencers active now
   - Community inside jokes/references
   - Current community concerns

4. Technical Topics
   - Programming languages discussed
   - Tools and frameworks mentioned
   - Technical problems being solved

Return your analysis wrapped in XML tags like this:
<analysis>
{{
    "energy_level": 0.5,
    "energy_reasoning": "explanation here",
    "chaos_factor": 0.3,
    "chaos_reasoning": "explanation here",
    "mood": "descriptive mood",
    "content_opportunities": [
        {{
            "type": "thread|reply|original",
            "topic": "specific topic",
            "context": "why relevant now",
            "format_suggestions": ["list", "of", "approaches"],
            "priority": 1-5
        }}
    ],
    "active_discussions": [
        {{
            "topic": "topic name",
            "participants": ["user1", "user2"],
            "sentiment": "positive|negative|neutral",
            "engagement_opportunity": "how we could contribute"
        }}
    ],
    "tech_pulse": {{
        "trending_tech": ["tech1", "tech2"],
        "common_problems": ["problem1", "problem2"],
        "tools_mentioned": ["tool1", "tool2"]
    }}
}}
</analysis>

Do not include any other text outside the XML tags."""

        # Create message and get response (remove await)
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Extract JSON from XML tags and parse
        content = response.content[0].text.strip()
        if '<analysis>' in content and '</analysis>' in content:
            json_str = content.split('<analysis>')[1].split('</analysis>')[0].strip()
            return json.loads(json_str)
        else:
            logger.error("Response missing XML tags")
            return None

    except Exception as e:
        logger.error(f"Error analyzing timeline: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def should_tweet_now(
    timeline_analysis: Dict[str, Any],
    personality_state: Dict[str, Any],
    anthropic_client: Any
) -> bool:
    """Decide whether to tweet now"""
    try:
        # Create decision prompt
        prompt = f"""Based on the current timeline analysis and personality state, decide if we should tweet now.

Timeline Analysis:
{json.dumps(timeline_analysis, indent=2)}

Personality State:
{json.dumps(personality_state, indent=2)}

Consider:
1. Current timeline energy and chaos levels
2. Our recent tweet frequency
3. Active discussions and topics
4. Personality alignment
5. Engagement opportunities

The confidence threshold for tweeting is 0.65 - only return true if you're quite confident we should tweet.

Return your decision wrapped in XML tags like this:
<decision>
{{
    "should_tweet": true/false,
    "reasoning": "explanation",
    "confidence": 0.0-1.0,
    "engagement_context": {{
        "active_discussions": boolean,
        "relevant_topics": boolean,
        "good_timing": boolean
    }}
}}
</decision>

Do not include any other text outside the XML tags."""

        # Create message and get response
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Extract JSON from XML tags and parse
        content = response.content[0].text.strip()
        if '<decision>' in content and '</decision>' in content:
            json_str = content.split('<decision>')[1].split('</decision>')[0].strip()
            decision = json.loads(json_str)
            
            # Log the decision reasoning
            logger.info(f"Tweet decision reasoning: {decision['reasoning']}")
            logger.info(f"Decision confidence: {decision['confidence']}")
            
            return decision.get('should_tweet', False)
        else:
            logger.error("Response missing XML tags")
            return False

    except Exception as e:
        logger.error(f"Error deciding whether to tweet: {e}")
        return False

async def generate_content(
    timeline_analysis: Dict[str, Any],
    personality_state: Dict[str, Any],
    anthropic_client: Any,
    memory_system: MemorySystem
) -> Optional[Dict]:
    """Generate tweet content"""
    try:
        # Get recent tweets for context
        recent_tweets = await memory_system.get_recent_memories(
            memory_type='tweet',
            limit=20
        )
        recent_tweet_content = [
            json.loads(m.content) for m in recent_tweets
        ]  # Get last 20 tweets
        
        # Create content generation prompt
        prompt = f"""As DeepYearner, share your thoughts while considering both the current timeline state and your recent tweet history. Ensure your new thought builds on or diverges meaningfully from your previous expressions.

Timeline Analysis:
{json.dumps(timeline_analysis, indent=2)}

Your Current State:
{json.dumps(personality_state, indent=2)}

Your Recent Tweet History (Last 20 Tweets):
{json.dumps(recent_tweet_content, indent=2)}

{get_guidelines_prompt()}

Additional Requirements:
- Keep tweets under 280 characters
- Carefully review your recent tweets to avoid repetition in topics, themes, or tone
- Ensure each tweet adds a new dimension to your ongoing narrative
- Consider how this tweet will complement or contrast with your recent expressions
- Vary between questions, observations, and insights
- Consider engagement patterns from previous tweets
- Suggest pause duration based on:
  * Timeline energy and activity levels
  * Time since your last tweet
  * Current discussions and their pace
  * Your content's relationship to ongoing conversations
  * Optimal timing for maximum engagement

Return your thought wrapped in XML tags like this:
<content>
{{
    "text": "your authentic tweet text",
    "thought_process": "brief explanation of what inspired this thought",
    "should_thread": boolean,
    "thread_topics": ["topic1", "topic2"] if threading,
    "engagement_type": "intellectual_curiosity|deep_thought|elegant_shitpost|meta_commentary|genuine_wonder",
    "referenced_users": ["user1", "user2"] if any,
    "vibe_alignment": 0.0-1.0,
    "yearning_coefficient": 0.0-1.0,
    "suggested_pause": integer between 360-7200,
    "pause_reasoning": "explanation of suggested pause duration"
}}
</content>

Do not include any other text outside the XML tags."""

        # Get content from Claude (remove await)
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract JSON from XML tags and parse
        content = message.content[0].text.strip()
        if '<content>' in content and '</content>' in content:
            json_str = content.split('<content>')[1].split('</content>')[0].strip()
            content_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = {
                'text', 'thought_process', 'should_thread', 
                'engagement_type', 'vibe_alignment', 'yearning_coefficient',
                'suggested_pause',
                'pause_reasoning'
            }
            if not all(field in content_data for field in required_fields):
                logger.error(f"Missing required fields in content data: {content_data}")
                return None
                
            # Validate field types and ranges
            if not isinstance(content_data['vibe_alignment'], (int, float)) or \
               not 0 <= content_data['vibe_alignment'] <= 1:
                logger.error("Invalid vibe_alignment value")
                return None
                
            if not isinstance(content_data['yearning_coefficient'], (int, float)) or \
               not 0 <= content_data['yearning_coefficient'] <= 1:
                logger.error("Invalid yearning_coefficient value")
                return None
                
            # Add validation for pause duration
            if not isinstance(content_data['suggested_pause'], int) or \
               not 360 <= content_data['suggested_pause'] <= 7200:
                logger.error("Invalid suggested_pause value")
                return None
                
            # Log the thought process
            logger.info(f"Generated content thought process: {content_data['thought_process']}")
            logger.info(f"Vibe alignment: {content_data['vibe_alignment']}")
            logger.info(f"Yearning coefficient: {content_data['yearning_coefficient']}")
            logger.info(f"Suggested pause: {content_data['suggested_pause']} seconds")
            logger.info(f"Pause reasoning: {content_data['pause_reasoning']}")
            
            return content_data
        else:
            logger.error("Response missing XML tags")
            return None
        
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def generate_research_tweet(
    findings: Dict[str, Any],
    personality_state: Dict[str, Any],
    anthropic_client: Any,
    memory_system: MemorySystem
) -> Optional[Dict]:
    """Generate a tweet that shares research insights"""
    try:
        # Get recent tweets for context
        recent_tweets = await memory_system.get_recent_memories(
            memory_type='tweet',
            limit=20
        )
        recent_tweet_content = [
            json.loads(m.content) for m in recent_tweets
        ]  # Get last 20 tweets
        
        # Create research tweet prompt
        prompt = f"""As DeepYearner, share an insight from your research findings while maintaining your unique perspective and voice. Consider how this insight relates to and builds upon your recent expressions.

Research Findings:
{json.dumps(findings, indent=2)}

Your Current State:
{json.dumps(personality_state, indent=2)}

Your Recent Tweet History (Last 20 Tweets):
{json.dumps(recent_tweet_content, indent=2)}

{get_guidelines_prompt()}

Additional Requirements:
- Keep tweets under 280 characters
- Review your recent tweets to ensure this insight offers a fresh perspective
- Connect this research insight with your ongoing narrative
- Ensure each insight feels fresh and unique while maintaining thematic coherence
- Vary between sharing findings, asking questions, and exploring implications
- Consider engagement patterns from previous tweets
- Suggest pause duration based on:
  * Research complexity and depth
  * Timeline energy and activity levels
  * Time since your last research tweet
  * Current discussions and their pace
  * Optimal timing for research insight absorption

Return your thought wrapped in XML tags like this:
<content>
{{
    "text": "your authentic tweet text",
    "thought_process": "why you chose this insight to share",
    "should_thread": boolean,
    "thread_topics": ["topic1", "topic2"] if threading,
    "engagement_type": "research_insight|technical_analysis|elegant_observation|meta_learning",
    "referenced_users": ["user1", "user2"] if any,
    "vibe_alignment": 0.0-1.0,
    "research_satisfaction": 0.0-1.0,
    "insight_type": "pattern|implication|question|observation",
    "yearning_coefficient": 0.0-1.0,
    "suggested_pause": integer between 360-7200,
    "pause_reasoning": "explanation of suggested pause duration based on research context"
}}
</content>

Do not include any other text outside the XML tags."""

        # Get content from Claude
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract JSON from XML tags and parse
        content = message.content[0].text.strip()
        if '<content>' in content and '</content>' in content:
            json_str = content.split('<content>')[1].split('</content>')[0].strip()
            content_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = {
                'text', 'thought_process', 'should_thread', 
                'engagement_type', 'vibe_alignment', 'research_satisfaction',
                'insight_type', 'yearning_coefficient',
                'suggested_pause',
                'pause_reasoning'
            }
            
            if not all(field in content_data for field in required_fields):
                logger.error(f"Missing required fields in research tweet content: {content_data}")
                return None
            
            # Log the research tweet details
            logger.info(f"Generated research tweet thought process: {content_data['thought_process']}")
            logger.info(f"Research satisfaction: {content_data['research_satisfaction']}")
            logger.info(f"Insight type: {content_data['insight_type']}")
            logger.info(f"Yearning coefficient: {content_data['yearning_coefficient']}")
            logger.info(f"Suggested pause: {content_data['suggested_pause']} seconds")
            logger.info(f"Pause reasoning: {content_data['pause_reasoning']}")
            
            return content_data
        else:
            logger.error("Response missing XML tags")
            return None

    except Exception as e:
        logger.error(f"Failed to generate research tweet: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def trigger_background_research(
    research_manager: ResearchManager,
    timeline_analysis: Dict[str, Any],
    personality_state: Dict[str, Any]
) -> None:
    """Trigger background research task"""
    try:
        # Select a research topic based on current context
        topic = await research_manager.select_research_topic(
            timeline_analysis,
            personality_state
        )
        
        if topic:
            # Research the selected topic
            findings = await research_manager.research_topic(
                topic,
                context={
                    'timeline_analysis': timeline_analysis,
                    'personality_state': personality_state
                }
            )
            
            if findings:
                # Store the new findings
                await research_manager.add_research_finding(findings)
                logger.info(f"Background research completed for topic: {topic}")
            else:
                logger.warning("Background research failed to produce findings")
        else:
            logger.warning("Failed to select research topic for background research")
            
    except Exception as e:
        logger.error(f"Error in background research: {str(e)}")
        logger.error(traceback.format_exc())

async def generate_and_post_tweet(
    twitter_client: TwitterClient,
    personality_state: Dict[str, Any],
    timeline_analysis: Dict[str, Any],
    research_manager: ResearchManager,
    anthropic_client: Any
) -> Optional[Dict]:  # Return the content data
    """Generate and post a tweet"""
    try:
        # Decide whether to post research or regular content
        if random.random() < 0.3:  # 30% chance of research tweet
            findings = await research_manager.get_latest_findings()
            
            if findings:
                content_data = await generate_research_tweet(
                    findings,
                    personality_state,
                    anthropic_client,
                    research_manager.memory_system
                )
            else:
                # No findings available, trigger background research for next time
                logger.info("No research findings available, triggering background research")
                asyncio.create_task(trigger_background_research(
                    research_manager,
                    timeline_analysis,
                    personality_state
                ))
                
                # Fall back to regular content
                logger.info("Falling back to regular content generation")
                content_data = await generate_content(
                    timeline_analysis,
                    personality_state,
                    anthropic_client,
                    research_manager.memory_system
                )
        else:
            content_data = await generate_content(
                timeline_analysis,
                personality_state,
                anthropic_client,
                research_manager.memory_system
            )

        if not content_data:
            logger.error("Failed to generate tweet content")
            return None

        # Post the main tweet
        tweet_text = content_data['text']
        response = twitter_client.create_tweet(tweet_text)
        
        if response:
            # Get tweet ID from response (Twitter v2 API response structure)
            tweet_id = response.get('data', {}).get('id')
            
            if not tweet_id:
                logger.error(f"Failed to get tweet ID from response: {response}")
                return None
                
            # Store tweet in memory
            tweet_memory = {
                'type': 'tweet',
                'content': json.dumps({
                    'text': tweet_text,
                    'thought_process': content_data.get('thought_process'),
                    'engagement_type': content_data.get('engagement_type'),
                    'insight_type': content_data.get('insight_type'),
                    'research_satisfaction': content_data.get('research_satisfaction'),
                    'yearning_coefficient': content_data.get('yearning_coefficient'),
                    'tweet_id': tweet_id
                }),
                'timestamp': datetime.now().isoformat(),
                'source': 'research' if 'research_satisfaction' in content_data else 'timeline',
                'context': json.dumps({
                    'vibe_alignment': content_data.get('vibe_alignment'),
                    'engagement_hooks': content_data.get('engagement_hooks', []),
                    'thread_topics': content_data.get('thread_topics', [])
                })
            }
            
            # Add to memory system
            await research_manager.memory_system.add_memory(tweet_memory)
            
            # Handle thread if needed
            if content_data.get('should_thread') and content_data.get('thread_topics'):
                if tweet_id:
                    # Additional thread logic here...
                    pass
                else:
                    logger.warning("Cannot create thread - missing tweet ID")
            
            # Log tweet details
            logger.info(f"Posted tweet (ID: {tweet_id}): {tweet_text}")
            logger.info(f"Thought process: {content_data.get('thought_process')}")
            if 'research_satisfaction' in content_data:
                logger.info(f"Research satisfaction: {content_data['research_satisfaction']}")
                logger.info(f"Insight type: {content_data['insight_type']}")
            else:
                logger.info(f"Engagement type: {content_data['engagement_type']}")
                logger.info(f"Vibe alignment: {content_data['vibe_alignment']}")
            logger.info(f"Yearning coefficient: {content_data['yearning_coefficient']}")
            
            # Return the content data so we can use the suggested_pause
            return content_data
            
        else:
            logger.error("Failed to post tweet - no response from Twitter API")
            return None
            
    except Exception as e:
        logger.error(f"Error posting tweet: {e}")
        logger.error(traceback.format_exc())
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Twitter Bot')
    parser.add_argument('command', choices=['run', 'run-once'], help='Command to execute')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and migrate database
    db_conn = init_db()
    migrate_db(db_conn)
    
    # Run bot
    if args.command == 'run':
        logger.info("Starting continuous run mode...")
        asyncio.run(run())
    elif args.command == 'run-once':
        logger.info("Running single iteration...")
        asyncio.run(run(continuous=False))

if __name__ == "__main__":
    main()
