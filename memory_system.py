from typing import Dict, List, Optional, Tuple, Set
import logging
import json
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

@dataclass
class Episode:
    """Represents a memory episode"""
    timestamp: datetime
    type: str
    content: Dict
    importance: float
    emotions: Dict[str, float]
    participants: Set[str]
    context: Dict
    references: List[str]  # IDs of related episodes
    embedding: Optional[np.ndarray] = None

class SemanticConcept:
    """Represents a semantic concept in memory"""
    def __init__(self, name: str):
        self.name = name
        self.connections: Dict[str, float] = {}  # Other concepts and their strengths
        self.episodes: List[str] = []  # Episode IDs where this concept appears
        self.last_updated = datetime.now()
        self.importance = 0.0
        self.embedding: Optional[np.ndarray] = None
        
    def update_connection(self, concept: str, strength: float):
        """Update connection strength with another concept"""
        current = self.connections.get(concept, 0.0)
        # Exponential moving average
        self.connections[concept] = current * 0.7 + strength * 0.3
        self.last_updated = datetime.now()

class Relationship:
    """Tracks relationship with another account"""
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.interactions: List[Dict] = []
        self.sentiment_history: List[Tuple[datetime, float]] = []
        self.trust_score = 0.5
        self.influence_score = 0.5
        self.last_interaction = None
        self.topics: Dict[str, float] = {}  # Topics and their relevance
        self.shared_interests: Set[str] = set()
        self.interaction_style = {
            'formality': 0.5,
            'humor': 0.5,
            'intellectuality': 0.5
        }
        
    def add_interaction(self, interaction: Dict):
        """Add new interaction and update relationship metrics"""
        self.interactions.append(interaction)
        self.last_interaction = datetime.now()
        
        # Update sentiment history
        if 'sentiment' in interaction:
            self.sentiment_history.append((
                datetime.now(),
                interaction['sentiment']
            ))
            
        # Update trust score
        if 'trust_impact' in interaction:
            self.trust_score = min(
                1.0,
                max(0.0, self.trust_score + interaction['trust_impact'])
            )
            
        # Update influence
        if 'metrics' in interaction:
            metrics = interaction['metrics']
            influence_delta = min(
                (
                    metrics.get('likes', 0) * 0.01 +
                    metrics.get('replies', 0) * 0.05 +
                    metrics.get('retweets', 0) * 0.03
                ),
                0.1
            )
            self.influence_score = min(
                1.0,
                self.influence_score + influence_delta
            )
            
        # Update topics
        if 'topics' in interaction:
            for topic, relevance in interaction['topics'].items():
                current = self.topics.get(topic, 0.0)
                self.topics[topic] = current * 0.8 + relevance * 0.2
                
        # Update interaction style
        if 'style' in interaction:
            style = interaction['style']
            for aspect in self.interaction_style:
                if aspect in style:
                    current = self.interaction_style[aspect]
                    self.interaction_style[aspect] = current * 0.8 + style[aspect] * 0.2

class MemorySystem:
    """Enhanced memory system with episodic and semantic memory"""
    
    def __init__(self, db: Any, vector_store: Any):
        self.db = db
        self.vector_store = vector_store
        self.episodes: Dict[str, Episode] = {}
        self.concepts: Dict[str, SemanticConcept] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.working_memory: List[str] = []  # Episode IDs
        self.attention_weights: Dict[str, float] = {}
        self.last_cleanup = datetime.now()
        self.last_consolidation = datetime.now()
        
    def add_episode(
        self,
        type: str,
        content: Dict,
        context: Dict,
        participants: Set[str]
    ) -> str:
        """Add new episode to memory"""
        # Generate embedding
        text_content = self._extract_text_content(content)
        embedding = self.vector_store.get_embedding(text_content)
        
        # Calculate importance
        importance = self._calculate_importance(content, context)
        
        # Extract emotions
        emotions = self._extract_emotions(content)
        
        # Find related episodes
        related = self._find_related_episodes(embedding, content)
        
        # Create episode
        episode = Episode(
            timestamp=datetime.now(),
            type=type,
            content=content,
            importance=importance,
            emotions=emotions,
            participants=participants,
            context=context,
            references=[e.id for e in related],
            embedding=embedding
        )
        
        # Generate ID
        episode_id = f"{type}_{datetime.now().timestamp()}"
        self.episodes[episode_id] = episode
        
        # Update working memory
        self._update_working_memory(episode_id)
        
        # Update semantic memory
        self._update_semantic_memory(episode)
        
        # Update relationships
        self._update_relationships(episode)
        
        # Store in database
        self.db["episodes"].insert({
            "id": episode_id,
            "type": type,
            "content": json.dumps(content),
            "context": json.dumps(context),
            "importance": importance,
            "emotions": json.dumps(emotions),
            "participants": list(participants),
            "timestamp": datetime.now().isoformat(),
            "references": episode.references
        })
        
        return episode_id
        
    def get_relevant_episodes(
        self,
        query: str,
        type: Optional[str] = None,
        limit: int = 5
    ) -> List[Episode]:
        """Get episodes relevant to query"""
        query_embedding = self.vector_store.get_embedding(query)
        
        relevant = []
        for episode_id, episode in self.episodes.items():
            if type and episode.type != type:
                continue
                
            if episode.embedding is not None:
                similarity = 1 - cosine(query_embedding, episode.embedding)
                relevant.append((similarity, episode))
                
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in relevant[:limit]]
        
    def get_relationship(self, account_id: str) -> Optional[Relationship]:
        """Get relationship with account"""
        return self.relationships.get(account_id)
        
    def get_related_concepts(
        self,
        concept: str,
        min_strength: float = 0.3
    ) -> List[Tuple[str, float]]:
        """Get concepts related to given concept"""
        if concept not in self.concepts:
            return []
            
        related = [
            (c, s) for c, s in self.concepts[concept].connections.items()
            if s >= min_strength
        ]
        related.sort(key=lambda x: x[1], reverse=True)
        return related
        
    def consolidate_memory(self):
        """Periodic memory consolidation"""
        if datetime.now() - self.last_consolidation < timedelta(hours=1):
            return
            
        # Update concept importance
        for concept in self.concepts.values():
            recent_episodes = len([
                e for e in concept.episodes
                if datetime.now() - self.episodes[e].timestamp < timedelta(days=7)
            ])
            concept.importance = min(
                recent_episodes * 0.1 +
                len(concept.connections) * 0.05,
                1.0
            )
            
        # Cleanup old episodes
        old_episodes = [
            eid for eid, e in self.episodes.items()
            if (
                datetime.now() - e.timestamp > timedelta(days=30) and
                e.importance < 0.3
            )
        ]
        for eid in old_episodes:
            del self.episodes[eid]
            
        # Update relationship scores
        for relationship in self.relationships.values():
            if relationship.last_interaction:
                days_since = (
                    datetime.now() - relationship.last_interaction
                ).days
                if days_since > 30:
                    relationship.trust_score *= 0.9
                    relationship.influence_score *= 0.9
                    
        self.last_consolidation = datetime.now()
        
    def _update_working_memory(self, episode_id: str):
        """Update working memory with new episode"""
        self.working_memory.append(episode_id)
        
        # Calculate attention weights
        total_weight = 0
        for eid in self.working_memory:
            episode = self.episodes[eid]
            recency = 1.0 / (
                1 + (datetime.now() - episode.timestamp).total_seconds() / 3600
            )
            self.attention_weights[eid] = (
                episode.importance * 0.6 + recency * 0.4
            )
            total_weight += self.attention_weights[eid]
            
        # Normalize weights
        for eid in self.working_memory:
            self.attention_weights[eid] /= total_weight
            
        # Keep only top K episodes
        if len(self.working_memory) > 10:
            sorted_episodes = sorted(
                self.working_memory,
                key=lambda x: self.attention_weights[x],
                reverse=True
            )
            self.working_memory = sorted_episodes[:10]
            
    def _update_semantic_memory(self, episode: Episode):
        """Update semantic memory with new episode"""
        # Extract concepts
        concepts = self._extract_concepts(episode.content)
        
        # Update concept network
        for concept, relevance in concepts.items():
            if concept not in self.concepts:
                self.concepts[concept] = SemanticConcept(concept)
                
            # Add episode reference
            self.concepts[concept].episodes.append(episode.id)
            
            # Update connections
            for other_concept, other_relevance in concepts.items():
                if other_concept != concept:
                    connection_strength = (
                        relevance * other_relevance
                    )
                    self.concepts[concept].update_connection(
                        other_concept,
                        connection_strength
                    )
                    
    def _update_relationships(self, episode: Episode):
        """Update relationships based on episode"""
        for participant in episode.participants:
            if participant not in self.relationships:
                self.relationships[participant] = Relationship(participant)
                
            # Add interaction
            self.relationships[participant].add_interaction({
                'type': episode.type,
                'content': episode.content,
                'sentiment': episode.emotions.get('sentiment', 0.0),
                'trust_impact': self._calculate_trust_impact(episode),
                'metrics': episode.content.get('metrics', {}),
                'topics': self._extract_concepts(episode.content),
                'style': self._extract_interaction_style(episode)
            })
            
    def _calculate_importance(self, content: Dict, context: Dict) -> float:
        """Calculate episode importance"""
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
                0.4
            )
            
        # Check emotional intensity
        if 'emotions' in content:
            emotions = content['emotions']
            importance += min(
                sum(abs(v) for v in emotions.values()) * 0.2,
                0.3
            )
            
        # Check novelty
        if 'embedding' in content:
            similar_episodes = self._find_related_episodes(
                content['embedding'],
                content
            )
            novelty = 1.0 - min(len(similar_episodes) * 0.1, 0.8)
            importance += novelty * 0.3
            
        return min(importance, 1.0)
        
    def _calculate_trust_impact(self, episode: Episode) -> float:
        """Calculate impact on trust score"""
        impact = 0.0
        
        # Positive interactions
        if episode.emotions.get('sentiment', 0.0) > 0.5:
            impact += 0.05
            
        # Quality engagement
        if 'metrics' in episode.content:
            metrics = episode.content['metrics']
            impact += min(
                (
                    metrics.get('likes', 0) * 0.005 +
                    metrics.get('replies', 0) * 0.01
                ),
                0.1
            )
            
        # Negative interactions
        if episode.emotions.get('sentiment', 0.0) < -0.5:
            impact -= 0.1
            
        return max(-0.2, min(impact, 0.2))
        
    def _extract_concepts(self, content: Dict) -> Dict[str, float]:
        """Extract concepts and their relevance from content"""
        concepts = {}
        
        # Extract from text
        if 'text' in content:
            # Use NLP to extract key concepts
            # For now, using simple word frequency
            words = content['text'].lower().split()
            for word in words:
                if len(word) > 3:  # Simple filter
                    concepts[word] = concepts.get(word, 0) + 1
                    
        # Extract from topics
        if 'topics' in content:
            for topic, relevance in content['topics'].items():
                concepts[topic] = max(
                    concepts.get(topic, 0),
                    relevance
                )
                
        # Normalize relevance scores
        if concepts:
            max_relevance = max(concepts.values())
            for concept in concepts:
                concepts[concept] /= max_relevance
                
        return concepts
        
    def _extract_emotions(self, content: Dict) -> Dict[str, float]:
        """Extract emotions from content"""
        emotions = {
            'sentiment': 0.0,
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0
        }
        
        # Use content's emotion scores if available
        if 'emotions' in content:
            emotions.update(content['emotions'])
            
        # Use sentiment if available
        if 'sentiment' in content:
            emotions['sentiment'] = content['sentiment']
            
        return emotions
        
    def _extract_interaction_style(self, episode: Episode) -> Dict[str, float]:
        """Extract interaction style metrics"""
        style = {
            'formality': 0.5,
            'humor': 0.0,
            'intellectuality': 0.5
        }
        
        if 'style' in episode.content:
            style.update(episode.content['style'])
            
        return style
        
    def _extract_text_content(self, content: Dict) -> str:
        """Extract text content for embedding"""
        text_parts = []
        
        if 'text' in content:
            text_parts.append(content['text'])
            
        if 'topics' in content:
            text_parts.extend(content['topics'].keys())
            
        if 'summary' in content:
            text_parts.append(content['summary'])
            
        return " ".join(text_parts)
        
    def _find_related_episodes(
        self,
        embedding: np.ndarray,
        content: Dict,
        threshold: float = 0.8
    ) -> List[Episode]:
        """Find related episodes"""
        related = []
        
        for episode in self.episodes.values():
            if episode.embedding is not None:
                similarity = 1 - cosine(embedding, episode.embedding)
                if similarity >= threshold:
                    related.append(episode)
                    
        return related
