"""
Event-driven personality system that manages bot's personality states and transitions.
Uses unified MemorySystem interface.
"""
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import json
import random
from enum import Enum, auto
from dataclasses import dataclass
import uuid

from memory_system import MemorySystem, Memory

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
    
    def __init__(self, memory_system: MemorySystem, anthropic_client=None):
        """Initialize personality system with unified memory system"""
        self.memory = memory_system
        self.current_state = self._initialize_state()
        self.event_queue: List[PersonalityEvent] = []
        self.last_transition = datetime.now()
        self.last_state_update = datetime.now()
        self.anthropic_client = anthropic_client
        
        # Initialize personality state in memory system
        self.memory.update_personality_state({
            'mood': self.current_state['mood'],
            'energy': self.current_state['energy'],
            'focus': self.current_state['focus'],
            'traits': self.current_state['traits']
        })
        
        # Transition triggers
        self.triggers = {
            'negativity_threshold': 0.7,  # High negativity triggers transition
            'success_threshold': 0.8,     # High success triggers transition
            'energy_threshold': 0.3,      # Low energy triggers transition
            'stress_threshold': 0.8,      # High stress triggers transition
            'min_time_between_transitions': timedelta(hours=1)
        }
    
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
            'interests': set(),
            'relationships': {},
            'goals': [],
            'values': {
                'authenticity': 0.9,
                'curiosity': 0.8,
                'helpfulness': 0.7
            }
        }
    
    def add_event(self, event: PersonalityEvent):
        """Add a new event that might influence personality"""
        try:
            # Add event to queue
            self.event_queue.append(event)
            
            # Store event in memory system
            memory = Memory(
                id=str(uuid.uuid4()),
                timestamp=event.timestamp,
                type='personality_event',
                content={
                    'event_type': event.type,
                    'intensity': event.intensity,
                    'sentiment': event.sentiment,
                    'metadata': event.metadata,
                    'source': event.source
                },
                context={
                    'mood': self.current_state['mood'],
                    'energy': self.current_state['energy'],
                    'focus': self.current_state['focus']
                }
            )
            self.memory.add_memory(memory)
            
            # Process events
            self._process_events()
            
        except Exception as e:
            logger.error(f"Error adding personality event: {str(e)}")
    
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
            # Update traits based on events
            for trait in MoodType:
                current = self.current_state['traits'][trait.name]
                adjustment = random.uniform(-0.1, 0.1)
                self.current_state['traits'][trait.name] = max(
                    0.1,
                    min(1.0, current + adjustment)
                )
            
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
            
    def get_state(self) -> Dict:
        """Get current personality state"""
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

    def generate_content(self, context: Dict) -> Optional[str]:
        """Generate content based on personality and context using Claude"""
        try:
            if not self.anthropic_client:
                logger.error("No Anthropic client available for content generation")
                return None

            # Get current mood and energy
            mood = self.current_state.get('mood', 'neutral')
            energy = self.current_state.get('energy', 0.5)
            
            # Create prompt based on personality state
            prompt = f"""You are a tech-focused Twitter bot with the following personality state:
Mood: {mood}
Energy Level: {energy}
Traits: {json.dumps(self.current_state.get('traits', {}))}

Generate a single tweet (max 280 chars) that:
1. Is relevant to tech, coding, or AI
2. Matches the current mood and energy level
3. Encourages engagement and discussion
4. Is authentic and thoughtful
5. Avoids controversy or negativity

Current context:
{json.dumps(context, indent=2)}

Generate only the tweet text, no other commentary."""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            tweet_text = response.content[0].text.strip()
            if len(tweet_text) > 280:
                tweet_text = tweet_text[:277] + "..."
                
            return tweet_text
                
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return None

    def generate_reply(self, tweet: Dict, context: Dict) -> Optional[str]:
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

Generate only the reply text, no other commentary."""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            reply_text = response.content[0].text.strip()
            if len(reply_text) > 280:
                reply_text = reply_text[:277] + "..."
                
            return reply_text
                
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

    def determine_sleep_duration(self, recent_activity: Dict) -> int:
        """Use Claude to determine how long to sleep based on recent activity"""
        try:
            if not self.anthropic_client:
                logger.error("No Anthropic client available")
                return 900  # Default to 15 minutes

            prompt = f"""As a Twitter bot, analyze my recent activity and recommend how long to sleep (in seconds) before the next iteration.

Recent Activity:
{json.dumps(recent_activity, indent=2)}

Consider:
1. Time since last tweet/reply/like/retweet
2. Current engagement levels
3. Time of day
4. Rate limit status
5. Activity patterns of my followers

Respond with ONLY a number representing seconds to sleep (between 300 and 3600).
Do not include any other text or explanation."""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            sleep_duration = int(response.content[0].text.strip())
            # Ensure sleep duration is within reasonable bounds
            return max(300, min(3600, sleep_duration))  # Between 5 minutes and 1 hour
                
        except Exception as e:
            logger.error(f"Error determining sleep duration: {str(e)}")
            return 900  # Default to 15 minutes if there's an error

    def should_tweet_now(self, context: Dict) -> bool:
        """Use Claude to decide whether to tweet based on current context"""
        try:
            if not self.anthropic_client:
                logger.error("No Anthropic client available")
                return False

            prompt = f"""As a Twitter bot, analyze the current context and decide if I should tweet now.

Current Context:
{json.dumps(context, indent=2)}

Personality State:
{json.dumps(self.current_state, indent=2)}

Consider:
1. Time since last tweet
2. Current engagement levels
3. Recent tweet performance
4. Mood and energy levels
5. Ongoing conversations
6. Time of day
7. Current trending topics

Respond with ONLY 'yes' or 'no'.
Do not include any other text or explanation."""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            decision = response.content[0].text.strip().lower()
            return decision == 'yes'
                
        except Exception as e:
            logger.error(f"Error deciding whether to tweet: {str(e)}")
            return False
