from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
import json
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from enum import Enum
import math

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

@dataclass
class PersonalityMode:
    """Represents the current personality mode with various dimensions"""
    blessed: float = 0.0
    cursed: float = 0.0
    intellectual: float = 0.0
    chaotic: float = 0.0
    sincere: float = 0.0
    ironic: float = 0.0
    analytical: float = 0.0
    playful: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'blessed': self.blessed,
            'cursed': self.cursed,
            'intellectual': self.intellectual,
            'chaotic': self.chaotic,
            'sincere': self.sincere,
            'ironic': self.ironic,
            'analytical': self.analytical,
            'playful': self.playful
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PersonalityMode':
        return cls(**data)
        
    def get_dominant_traits(self, threshold: float = 0.6) -> List[MoodType]:
        """Get list of dominant personality traits"""
        traits = []
        for trait, value in self.to_dict().items():
            if value >= threshold:
                traits.append(MoodType(trait))
        return traits
        
    def blend_with(self, other: 'PersonalityMode', factor: float) -> 'PersonalityMode':
        """Blend this personality mode with another"""
        result = PersonalityMode()
        for trait in MoodType:
            current = getattr(self, trait.value)
            target = getattr(other, trait.value)
            blended = current + (target - current) * factor
            setattr(result, trait.value, blended)
        return result

class EnergyState:
    """Manages the bot's energy levels and activity patterns"""
    
    def __init__(self):
        self.mental_energy: float = 1.0
        self.social_energy: float = 1.0
        self.creative_energy: float = 1.0
        self.last_rest = datetime.now()
        self.activity_history: List[Dict] = []
        
    def update_energy(self, activity_type: str, intensity: float):
        """Update energy levels based on activity"""
        if activity_type == 'social':
            self.social_energy -= intensity * 0.2
        elif activity_type == 'creative':
            self.creative_energy -= intensity * 0.15
        elif activity_type == 'intellectual':
            self.mental_energy -= intensity * 0.25
            
        # Natural recovery over time
        time_since_rest = (datetime.now() - self.last_rest).total_seconds() / 3600
        recovery = min(time_since_rest * 0.1, 0.5)
        
        self.mental_energy = min(1.0, self.mental_energy + recovery)
        self.social_energy = min(1.0, self.social_energy + recovery)
        self.creative_energy = min(1.0, self.creative_energy + recovery)
        
    def should_rest(self) -> bool:
        """Determine if bot needs to rest"""
        return (
            self.mental_energy < 0.3 or
            self.social_energy < 0.3 or
            self.creative_energy < 0.3
        )
        
    def take_rest(self):
        """Rest to recover energy"""
        self.last_rest = datetime.now()
        self.mental_energy = min(1.0, self.mental_energy + 0.4)
        self.social_energy = min(1.0, self.social_energy + 0.4)
        self.creative_energy = min(1.0, self.creative_energy + 0.4)

class MemoryState:
    """Manages the bot's memory and learning"""
    
    def __init__(self, db: Any):
        self.db = db
        self.short_term: List[Dict] = []
        self.working_memory: Dict[str, Any] = {}
        self.last_cleanup = datetime.now()
        
    def add_memory(self, memory_type: str, content: Dict):
        """Add a new memory"""
        timestamp = datetime.now().isoformat()
        
        # Add to short-term memory
        self.short_term.append({
            'type': memory_type,
            'content': content,
            'timestamp': timestamp
        })
        
        # Store in database
        self.db["memories"].insert({
            "type": memory_type,
            "content": json.dumps(content),
            "timestamp": timestamp,
            "importance": self._calculate_importance(content)
        })
        
    def get_recent_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent memories of specified type"""
        query = "SELECT * FROM memories"
        params = []
        
        if memory_type:
            query += " WHERE type = ?"
            params.append(memory_type)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        return list(self.db["memories"].rows_where(query, params))
        
    def cleanup_memory(self):
        """Cleanup old memories"""
        if datetime.now() - self.last_cleanup < timedelta(hours=1):
            return
            
        # Clear old short-term memories
        self.short_term = [
            m for m in self.short_term
            if datetime.fromisoformat(m['timestamp']) > datetime.now() - timedelta(hours=24)
        ]
        
        # Clear old working memory
        for key in list(self.working_memory.keys()):
            if self.working_memory[key].get('timestamp'):
                if datetime.fromisoformat(
                    self.working_memory[key]['timestamp']
                ) < datetime.now() - timedelta(days=7):
                    del self.working_memory[key]
                    
        self.last_cleanup = datetime.now()
        
    def _calculate_importance(self, content: Dict) -> float:
        """Calculate memory importance score"""
        importance = 0.0
        
        # Check engagement metrics
        if 'metrics' in content:
            metrics = content['metrics']
            importance += min(
                (
                    metrics.get('likes', 0) * 0.01 +
                    metrics.get('replies', 0) * 0.05 +
                    metrics.get('retweets', 0) * 0.03
                ),
                1.0
            )
            
        # Check sentiment
        if 'sentiment' in content:
            sentiment = abs(content['sentiment'])
            importance += min(sentiment * 0.3, 0.3)
            
        # Check novelty
        if 'novelty_score' in content:
            importance += min(content['novelty_score'] * 0.2, 0.2)
            
        return min(importance, 1.0)

class PersonalitySystem:
    """Enhanced personality system with mood transitions and energy management"""
    
    def __init__(self, db: Any):
        self.current_mode = PersonalityMode(
            intellectual=0.5,
            sincere=0.5
        )
        self.target_mode = None
        self.transition_start = None
        self.transition_duration = None
        self.energy = EnergyState()
        self.memory = MemoryState(db)
        self.last_transition = datetime.now()
        self.transition_cooldown = timedelta(minutes=30)
        
    def update_from_timeline(self, timeline_analysis: Dict):
        """Update personality based on timeline analysis"""
        # Extract timeline mood
        timeline_mode = PersonalityMode(
            blessed=timeline_analysis.get('blessed', 0.0),
            cursed=timeline_analysis.get('cursed', 0.0),
            intellectual=timeline_analysis.get('intellectual', 0.0),
            chaotic=timeline_analysis.get('chaos', 0.0),
            sincere=timeline_analysis.get('sincerity', 0.0),
            ironic=timeline_analysis.get('irony', 0.0),
            analytical=timeline_analysis.get('analysis', 0.0),
            playful=timeline_analysis.get('playful', 0.0)
        )
        
        # Blend with current mode
        adaptation_rate = self._calculate_adaptation_rate()
        self.current_mode = self.current_mode.blend_with(
            timeline_mode,
            adaptation_rate
        )
        
    def update_from_interactions(self, interactions: List[Dict]):
        """Update personality based on recent interactions"""
        if not interactions:
            return
            
        # Calculate average interaction mood
        interaction_moods = []
        for interaction in interactions:
            if 'mood' in interaction:
                interaction_moods.append(PersonalityMode.from_dict(
                    interaction['mood']
                ))
                
        if interaction_moods:
            avg_mood = PersonalityMode()
            for trait in MoodType:
                total = sum(
                    getattr(mood, trait.value)
                    for mood in interaction_moods
                )
                setattr(
                    avg_mood,
                    trait.value,
                    total / len(interaction_moods)
                )
                
            # Blend with current mode
            self.current_mode = self.current_mode.blend_with(avg_mood, 0.3)
            
        # Update energy based on interactions
        for interaction in interactions:
            self.energy.update_energy(
                interaction.get('type', 'social'),
                interaction.get('intensity', 0.5)
            )
            
    def update_from_discoveries(self, discoveries: List[Dict]):
        """Update personality based on content discoveries"""
        if not discoveries:
            return
            
        # Calculate discovery impact
        discovery_mode = PersonalityMode()
        for discovery in discoveries:
            if discovery['type'] == 'trend':
                discovery_mode.chaotic += 0.2
                discovery_mode.playful += 0.1
            elif discovery['type'] == 'conversation':
                discovery_mode.social += 0.2
                discovery_mode.sincere += 0.1
            elif discovery['type'] == 'search_result':
                discovery_mode.intellectual += 0.2
                discovery_mode.analytical += 0.1
                
        # Normalize values
        for trait in MoodType:
            value = getattr(discovery_mode, trait.value)
            setattr(discovery_mode, trait.value, min(value, 1.0))
            
        # Blend with current mode
        self.current_mode = self.current_mode.blend_with(discovery_mode, 0.2)
        
    def should_transition(self) -> bool:
        """Determine if personality should transition"""
        if self.transition_start:
            return False
            
        if datetime.now() - self.last_transition < self.transition_cooldown:
            return False
            
        # Check energy levels
        if self.energy.should_rest():
            return True
            
        # Check dominant trait duration
        dominant_traits = self.current_mode.get_dominant_traits()
        if dominant_traits:
            trait_duration = (
                datetime.now() - self.last_transition
            ).total_seconds() / 3600
            if trait_duration > 2:  # 2 hours max in one dominant trait
                return True
                
        return random.random() < 0.1  # 10% random chance
        
    def start_transition(self, target_mode: PersonalityMode):
        """Start transition to new personality mode"""
        self.target_mode = target_mode
        self.transition_start = datetime.now()
        
        # Calculate transition duration based on difference
        total_diff = sum(
            abs(
                getattr(self.current_mode, trait.value) -
                getattr(target_mode, trait.value)
            )
            for trait in MoodType
        )
        
        # Duration between 10-30 minutes based on difference
        self.transition_duration = timedelta(
            minutes=max(10, min(30, total_diff * 20))
        )
        
    def update_transition(self):
        """Update personality transition"""
        if not self.transition_start or not self.target_mode:
            return
            
        # Calculate transition progress
        elapsed = datetime.now() - self.transition_start
        progress = min(
            elapsed.total_seconds() /
            self.transition_duration.total_seconds(),
            1.0
        )
        
        # Apply transition
        self.current_mode = self.current_mode.blend_with(
            self.target_mode,
            progress
        )
        
        # Check if transition complete
        if progress >= 1.0:
            self.transition_start = None
            self.target_mode = None
            self.last_transition = datetime.now()
            
    def _calculate_adaptation_rate(self) -> float:
        """Calculate rate of adaptation to timeline mood"""
        # Base rate
        rate = 0.2
        
        # Adjust based on energy
        if self.energy.mental_energy < 0.5:
            rate *= 0.7
        if self.energy.social_energy < 0.5:
            rate *= 0.8
            
        # Adjust based on current traits
        if self.current_mode.chaotic > 0.7:
            rate *= 1.3
        if self.current_mode.analytical > 0.7:
            rate *= 0.8
            
        return min(rate, 0.5)  # Cap at 0.5
        
    def get_state_summary(self) -> Dict:
        """Get summary of current personality state"""
        return {
            'mode': self.current_mode.to_dict(),
            'energy': {
                'mental': self.energy.mental_energy,
                'social': self.energy.social_energy,
                'creative': self.energy.creative_energy
            },
            'transitioning': bool(self.transition_start),
            'dominant_traits': [
                t.value for t in self.current_mode.get_dominant_traits()
            ]
        }
