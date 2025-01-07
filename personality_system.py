"""
Event-driven personality system that manages bot's personality states and transitions.
Uses unified MemorySystem interface.
"""
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
import json
import random
from enum import Enum, auto
from dataclasses import dataclass
import uuid
import os

from memory_system import MemorySystem, Memory
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class MoodType(Enum):
    BLESSED = "blessed"
    CURSED = "cursed"
    INTELLECTUAL = "intellectual"
    CHAOTIC = "chaotic"
    SINCERE = "sincere"
    IRONIC = "ironic"
    ANALYTICAL = "analytical"
    PLAYFUL = "playful"

class EmotionalState(Enum):
    """Current emotional state"""
    EXCITED = auto()
    FOCUSED = auto()
    RELAXED = auto()
    TIRED = auto()
    STRESSED = auto()

@dataclass
class PersonalityEvent:
    """Represents an event that might trigger personality changes"""
    type: str  # interaction, achievement, timeline, error
    timestamp: datetime
    intensity: float  # 0.0 to 1.0
    sentiment: float  # -1.0 to 1.0
    metadata: Dict
    source: str

class PersonalityMode:
    """Represents different personality modes with their relative strengths"""
    intellectual: float = 0.0
    poasting: float = 0.0
    yearning: float = 0.0
    blessed: float = 0.0
    cursed: float = 0.0

class PersonalitySystem:
    """Manages bot personality and state transitions"""
    
    def __init__(self, memory_system: MemorySystem, anthropic_client: Any):
        """Initialize personality system with unified memory system"""
        self.memory_system = memory_system
        self.anthropic_client = anthropic_client
        self.current_state = {}

    async def add_event(self, event: Dict[str, Any]) -> None:
        """Add an event to the personality system"""
        try:
            # Validate event structure
            required_fields = {'type', 'data', 'timestamp', 'source'}
            if not all(field in event for field in required_fields):
                raise ValueError(f"Event missing required fields: {required_fields - event.keys()}")

            # Process the event
            if event['type'] == 'timeline_analysis':
                # Update current state based on timeline analysis
                self.current_state.update({
                    'last_timeline_analysis': event['data'],
                    'last_timeline_timestamp': event['timestamp'],
                    'timeline_context': event.get('context', {})
                })

            # Store event in memory system
            await self.memory_system.add_memory({
                'type': 'event',
                'content': json.dumps(event['data']),
                'timestamp': event['timestamp'],
                'context': json.dumps(event.get('context', {})),
                'source': event['source']
            })

        except Exception as e:
            logger.error(f"Error adding personality event: {str(e)}")
            raise
    
    def _initialize_state(self) -> Dict:
        """Initialize personality state"""
        return {
            'mood': 'neutral',
            'energy': 1.0,
            'focus': 1.0,
            'traits': {
                'openness': 0.8,
                'conscientiousness': 0.7,
                'extraversion': 0.6,
                'agreeableness': 0.7,
                'neuroticism': 0.4
            },
            'interests': list(),
            'relationships': {},
            'goals': [],
            'values': {
                'authenticity': 0.9,
                'curiosity': 0.8,
                'helpfulness': 0.7
            }
        }
    
    def _process_events(self):
        """Process queued events and update state"""
        try:
            if not self.event_queue:
                return
            
            # Calculate aggregate metrics
            negativity = 0.0
            success = 0.0
            stress = 0.0
            total_events = len(self.event_queue)
            
            for event in self.event_queue:
                # Update metrics based on event type and sentiment
                if event.sentiment < 0:
                    negativity += abs(event.sentiment) * event.intensity
                if event.type in ['achievement', 'positive_interaction']:
                    success += event.intensity
                if event.type in ['conflict', 'pressure', 'deadline']:
                    stress += event.intensity
            
            # Normalize metrics
            if total_events > 0:
                negativity /= total_events
                success /= total_events
                stress /= total_events
            
            # Update state
            self._update_state(negativity, success, stress)
            
            # Clear queue
            self.event_queue.clear()
            
        except Exception as e:
            logger.error(f"Error processing personality events: {str(e)}")
    
    def _update_state(self, negativity: float, success: float, stress: float):
        """Update personality state based on events"""
        try:
            # Update energy based on stress and success
            energy_change = (success * 0.2) - (stress * 0.3)
            self.current_state['energy'] = max(0.1, min(1.0, 
                self.current_state['energy'] + energy_change))
            
            # Update focus based on stress
            focus_change = -0.2 if stress > self.triggers['stress_threshold'] else 0.1
            self.current_state['focus'] = max(0.1, min(1.0,
                self.current_state['focus'] + focus_change))
            
            # Update traits based on events
            self._update_traits(negativity, success, stress)
            
            # Try transition if conditions are met
            self._try_transition()
            
            # Update memory system with new state
            self.memory.update_personality_state({
                'mood': self.current_state['mood'],
                'energy': self.current_state['energy'],
                'focus': self.current_state['focus'],
                'traits': self.current_state['traits'],
                'last_update': datetime.now().isoformat()
            })
            
            self.last_state_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating personality state: {str(e)}")
    
    def _update_traits(self, negativity: float, success: float, stress: float):
        """Update personality traits based on events"""
        try:
            # Update base traits
            traits = self.current_state['traits']
            
            # Adjust openness based on success
            traits['openness'] = max(0.1, min(1.0, 
                traits.get('openness', 0.5) + (success * 0.1)))
            
            # Adjust conscientiousness based on stress
            traits['conscientiousness'] = max(0.1, min(1.0,
                traits.get('conscientiousness', 0.5) - (stress * 0.1)))
            
            # Adjust extraversion based on success and negativity
            traits['extraversion'] = max(0.1, min(1.0,
                traits.get('extraversion', 0.5) + (success * 0.1) - (negativity * 0.1)))
            
            # Adjust agreeableness based on negativity
            traits['agreeableness'] = max(0.1, min(1.0,
                traits.get('agreeableness', 0.5) - (negativity * 0.2)))
            
            # Adjust neuroticism based on stress
            traits['neuroticism'] = max(0.1, min(1.0,
                traits.get('neuroticism', 0.5) + (stress * 0.2)))
            
        except Exception as e:
            logger.error(f"Error updating personality traits: {str(e)}")
    
    def _try_transition(self):
        """Attempt to transition to a new personality state"""
        try:
            now = datetime.now()
            
            # Check if enough time has passed
            if (now - self.last_transition) < self.triggers['min_time_between_transitions']:
                return
            
            # Get recent memories for context
            recent_memories = self.memory.get_recent_memories(limit=20)
            
            # Choose new emotional state based on context
            available_states = list(EmotionalState)
            current_state = EmotionalState[self.current_state['emotional_state']]
            available_states.remove(current_state)
            
            # Weight states based on current metrics
            weights = []
            for state in available_states:
                weight = 1.0
                
                if state == EmotionalState.EXCITED:
                    weight *= self.current_state['energy']
                elif state == EmotionalState.FOCUSED:
                    weight *= self.current_state['focus']
                elif state == EmotionalState.RELAXED:
                    weight *= (1.0 - self.current_state['stress'])
                elif state == EmotionalState.TIRED:
                    weight *= (1.0 - self.current_state['energy'])
                    
                weights.append(weight)
                
            # Choose new state
            new_state = random.choices(available_states, weights=weights)[0]
            self.current_state['emotional_state'] = new_state.name
            self.last_transition = now
            
        except Exception as e:
            logger.error(f"Error in personality transition: {str(e)}")
            
    async def get_state(self) -> Dict[str, Any]:
        """Get current personality state"""
        try:
            # Get recent memories to inform state
            recent_memories = await self.memory_system.get_recent_memories(hours=24)
            
            # Update state with recent memory context
            state = self.current_state.copy()
            state.update({
                'recent_memory_count': len(recent_memories),
                'last_update': datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting personality state: {str(e)}")
            return self.current_state.copy()
    
    def get_trait(self, trait: MoodType) -> float:
        """Get specific personality trait value"""
        return self.current_state['traits'].get(trait.name, 0.5)
    
    def get_emotional_state(self) -> EmotionalState:
        """Get current emotional state"""
        return EmotionalState[self.current_state['emotional_state']]
    
    def get_current_state(self) -> Dict:
        """Get current personality state"""
        return {
            'mode': self.current_state.get('mood', 'neutral'),
            'energy': self.current_state.get('energy', 1.0),
            'dominant_traits': [trait for trait, value in 
                              self.current_state.get('traits', {}).items() 
                              if value > 0.7]
        }

    def should_engage(self, tweet: Dict, context: Dict) -> bool:
        """Decide whether to engage with a tweet based on personality and context"""
        try:
            # Get tweet metrics
            metrics = tweet.get('public_metrics', {})
            likes = metrics.get('like_count', 0)
            replies = metrics.get('reply_count', 0)
            retweets = metrics.get('retweet_count', 0)
            
            # Calculate engagement score
            engagement_score = (likes + replies * 2 + retweets) / 100
            
            # Get personality factors
            energy = self.current_state.get('energy', 0.5)
            extraversion = self.current_state.get('traits', {}).get('extraversion', 0.5)
            
            # Add randomization to reduce frequency of engagement
            if random.random() > 0.3:  # Only engage 30% of the time
                return False
            
            # Calculate engagement threshold
            threshold = 0.3 - (energy * 0.2) - (extraversion * 0.2)
            
            # Check mood context
            mood_context = context.get('mood_context', {})
            if mood_context.get('mood') == 'subdued':
                threshold += 0.2
            elif mood_context.get('mood') == 'excited':
                threshold -= 0.2
            
            # Don't engage with tweets that already have lots of replies
            if replies > 10:
                return False
                
            return engagement_score > threshold
            
        except Exception as e:
            logger.error(f"Error deciding engagement: {str(e)}")
            return False

    async def generate_content(self, context):
        """Generate content using Claude"""
        try:
            if not self.anthropic_client:
                logger.error("No Anthropic client available for content generation")
                return None
            
            # Get personality factors
            mood = self.current_state.get('mood', 'neutral')
            energy = self.current_state.get('energy', 0.5)
            traits = self.current_state.get('traits', {})
            
            prompt = f"""You are a tech-focused Twitter bot with the following personality state:
Mood: {mood}
Energy Level: {energy}
Traits: {json.dumps(traits)}

Generate a tweet that:
1. Matches the current mood and energy level
2. Aligns with our traits and interests
3. Is engaging and authentic
4. Stays under 280 characters
5. Avoids controversy or negativity

Current context:
{json.dumps(context, indent=2)}

Respond with ONLY an XML tag containing the tweet text.
Format your response exactly like this:
<tweet_content>Your tweet text here</tweet_content>

Do not include any other text or explanation."""

            # Get content from Claude
            message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Extract tweet content from response
            response_text = message.content[0].text
            tweet_content = response_text.split('<tweet_content>')[1].split('</tweet_content>')[0].strip()
            
            return tweet_content
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return None

    async def generate_reply(self, tweet: Dict, context: Dict) -> Optional[str]:
        """Generate a reply to a tweet using Claude"""
        try:
            if not self.anthropic_client:
                logger.error("No Anthropic client available for reply generation")
                return None

            # Get tweet content and personality factors
            tweet_text = tweet.get('text', '')
            mood = self.current_state.get('mood', 'neutral')
            energy = self.current_state.get('energy', 0.5)
            
            prompt = f"""You are a tech-focused Twitter bot with the following personality state:
Mood: {mood}
Energy Level: {energy}
Traits: {json.dumps(self.current_state.get('traits', {}))}

Generate a reply to this tweet:
"{tweet_text}"

The reply should:
1. Be relevant and add value to the conversation
2. Match the current mood and energy level
3. Be engaging but not overly familiar
4. Stay under 280 characters
5. Be authentic and thoughtful
6. Avoid controversy or negativity

Current context:
{json.dumps(context, indent=2)}

Respond with ONLY an XML tag containing the reply text.
Format your response exactly like this:
<reply_content>Your reply text here</reply_content>

Do not include any other text or explanation."""

            message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            try:
                response_text = message.content[0].text.strip()
                if '<reply_content>' in response_text and '</reply_content>' in response_text:
                    reply_text = response_text.split('<reply_content>')[1].split('</reply_content>')[0].strip()
                    return reply_text
                else:
                    logger.error("Claude response did not contain proper XML tags")
                    return None
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Failed to parse reply from Claude response: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating reply: {str(e)}")
            return None

    def calculate_follow_score(self, user_info: Dict) -> float:
        """Calculate a follow score for a user based on their info and our personality"""
        try:
            score = 0.5  # Base score
            
            # Get personality factors
            traits = self.current_state.get('traits', {})
            openness = traits.get('openness', 0.5)
            agreeableness = traits.get('agreeableness', 0.5)
            
            # Analyze user bio
            bio = user_info.get('description', '').lower()
            if any(interest in bio for interest in self.current_state.get('interests', [])):
                score += 0.2
                
            # Check metrics
            metrics = user_info.get('public_metrics', {})
            followers = metrics.get('followers_count', 0)
            following = metrics.get('following_count', 0)
            
            # Prefer balanced follow ratios
            if following > 0:
                ratio = followers / following
                if 0.5 <= ratio <= 2.0:
                    score += 0.1
                    
            # Adjust based on personality
            score += (openness * 0.1)  # More likely to follow if we're open
            score += (agreeableness * 0.1)  # More likely to follow if we're agreeable
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating follow score: {str(e)}")
            return 0.0

    async def determine_sleep_duration(self, recent_activity: Dict) -> int:
        """Use Claude to determine how long to sleep based on recent activity"""
        try:
            if not self.anthropic_client:
                logger.error("No Anthropic client available")
                return 900  # Default to 15 minutes

            # Format recent activity for better decision making
            formatted_activity = {
                'last_tweet_time': recent_activity.get('last_tweet', {}).get('created_at') if recent_activity.get('last_tweet') else None,
                'last_reply_time': recent_activity.get('last_reply', {}).get('created_at') if recent_activity.get('last_reply') else None,
                'last_retweet_time': recent_activity.get('last_retweet', {}).get('created_at') if recent_activity.get('last_retweet') else None,
                'last_quote_time': recent_activity.get('last_quote', {}).get('created_at') if recent_activity.get('last_quote') else None,
                'current_hour': datetime.now().hour,
                'current_mood': self.current_state.get('mood', 'neutral'),
                'current_energy': self.current_state.get('energy', 0.5),
                'rate_limited': any(v is None for v in recent_activity.values())
            }

            prompt = f"""Based on recent activity and current state, determine how long to sleep before next action.

Recent Activity:
{json.dumps(formatted_activity, indent=2)}

Consider:
1. Time since last action
2. Current hour (for circadian rhythm)
3. Rate limit status
4. Energy level
5. Mood

Return duration in seconds as an XML tag.
Format your response exactly like this:
<sleep_duration>900</sleep_duration>

Guidelines:
- Minimum: 300 seconds (5 minutes)
- Maximum: 3600 seconds (1 hour)
- If rate limited: at least 900 seconds
- Late night (11pm-6am): longer durations
- High energy: shorter durations
- Low energy: longer durations

Do not include any other text or explanation."""

            message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            try:
                response_text = message.content[0].text.strip()
                # Extract duration from XML tag
                if '<sleep_duration>' in response_text and '</sleep_duration>' in response_text:
                    duration = int(response_text.split('<sleep_duration>')[1].split('</sleep_duration>')[0])
                    return max(300, min(duration, 3600))  # Clamp between 5 minutes and 1 hour
                else:
                    logger.error("Claude response did not contain proper XML tags")
                    return 900
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Failed to parse sleep duration from Claude response: {e}")
                return 900

        except Exception as e:
            logger.error(f"Error determining sleep duration: {e}")
            return 900

    async def should_tweet_now(self, context: Dict) -> bool:
        """Use Claude to decide whether to tweet based on current context"""
        try:
            if not self.anthropic_client:
                logger.error("No Anthropic client available")
                return False

            # Get timeline state and check tweet frequency
            timeline_analysis = context.get('timeline_analysis', {})
            timeline_state = timeline_analysis.get('timeline_state', {})
            
            # If we've tweeted too much today, don't tweet
            if timeline_state.get('should_wait', False):
                logger.info(f"Not tweeting - already sent {timeline_state.get('our_tweets_24h', 0)} tweets in last 24h")
                return False

            # Convert sets to lists for JSON serialization
            state_copy = self.current_state.copy()
            if isinstance(state_copy.get('interests'), set):
                state_copy['interests'] = list(state_copy['interests'])

            prompt = f"""As a Twitter bot, analyze the current context and decide if I should tweet now.

Current Context:
{json.dumps(context, indent=2)}

Personality State:
{json.dumps(state_copy, indent=2)}

Consider:
1. Time since last tweet
2. Current engagement levels
3. Recent tweet performance
4. Mood and energy levels
5. Ongoing conversations
6. Time of day
7. Current trending topics
8. Our tweet frequency (aim for 8-12 tweets per day)

Important factors:
- Tweet if post opportunity is above 0.1
- Don't tweet if we've already posted {timeline_state.get('our_tweets_24h', 0)} tweets in last 24h
- Consider joining active discussions in timeline

Respond with ONLY an XML tag containing 'yes' or 'no'.
Format your response exactly like this:
<should_tweet>yes</should_tweet>
or
<should_tweet>no</should_tweet>

Do not include any other text or explanation."""

            # Add detailed debug logging for the prompt
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug("\n" + "="*50)
                logger.debug("SHOULD TWEET DECISION")
                logger.debug("="*50)
                logger.debug("\nPROMPT:")
                logger.debug("-"*50)
                logger.debug(prompt)
                logger.debug("-"*50)

            message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            try:
                response_text = message.content[0].text.strip()
                # Add detailed debug logging for the response
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    logger.debug("\nRESPONSE:")
                    logger.debug("-"*50)
                    logger.debug(response_text)
                    logger.debug("-"*50)
                    logger.debug("\n")

                # Extract decision from XML tag
                if '<should_tweet>' in response_text and '</should_tweet>' in response_text:
                    decision = response_text.split('<should_tweet>')[1].split('</should_tweet>')[0].lower()
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        logger.debug(f"DECISION: {decision}")
                    return decision == 'yes'
                else:
                    logger.error("Claude response did not contain proper XML tags")
                    return False
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Failed to parse tweet decision from Claude response: {e}")
                return False

        except Exception as e:
            logger.error(f"Error deciding whether to tweet: {e}")
            return False
