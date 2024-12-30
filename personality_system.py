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

class PersonalitySystem:
    """Manages bot personality and state transitions"""
    
    def __init__(self, memory_system: MemorySystem):
        """Initialize personality system with unified memory system"""
        self.memory = memory_system
        self.current_state = self._initialize_state()
        self.event_queue: List[PersonalityEvent] = []
        self.last_transition = datetime.now()
        self.last_state_update = datetime.now()
        
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
            'traits': {
                MoodType.BLESSED.name: random.uniform(0.5, 0.8),
                MoodType.CURSED.name: random.uniform(0.2, 0.5),
                MoodType.INTELLECTUAL.name: random.uniform(0.6, 0.9),
                MoodType.CHAOTIC.name: random.uniform(0.4, 0.7),
                MoodType.SINCERE.name: random.uniform(0.6, 0.8),
                MoodType.IRONIC.name: random.uniform(0.2, 0.5),
                MoodType.ANALYTICAL.name: random.uniform(0.6, 0.9),
                MoodType.PLAYFUL.name: random.uniform(0.4, 0.7)
            },
            'emotional_state': EmotionalState.FOCUSED.name,
            'energy': 1.0,
            'focus': 0.8,
            'stress': 0.2,
            'last_achievements': [],
            'active_topics': set(),
            'conversation_sentiment': 0.0,
            'timeline_mood': 0.0
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
                elif state == EmotionalState.STRESSED:
                    weight *= self.current_state['stress']
                
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
            
            # Choose new state
            new_state = random.choices(available_states, weights=weights)[0]
            
            # Update state
            self.current_state['emotional_state'] = new_state.name
            
            # Store transition in memory
            self.memory.add_memory(Memory(
                id=f"transition_{now.timestamp()}",
                timestamp=now,
                type='personality_transition',
                content={
                    'previous_state': current_state.name,
                    'new_state': new_state.name,
                    'reason': 'event_triggered'
                },
                context=self.current_state
            ))
            
            self.last_transition = now
            
            logger.info(
                f"Personality transitioned from {current_state.name} to {new_state.name}"
            )
            
        except Exception as e:
            logger.error(f"Error during personality transition: {str(e)}")
    
    def get_state(self) -> Dict:
        """Get current personality state"""
        return self.current_state.copy()
    
    def get_trait(self, trait: MoodType) -> float:
        """Get specific personality trait value"""
        return self.current_state['traits'].get(trait.name, 0.5)
    
    def get_emotional_state(self) -> EmotionalState:
        """Get current emotional state"""
        return EmotionalState[self.current_state['emotional_state']]
