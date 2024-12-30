"""
Unified memory system that handles all types of memory including personality state.
Provides a centralized interface for storing and retrieving memories, experiences,
and state information.
"""
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import numpy as np
from vector_store import VectorStore, VectorDocument

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """Base class for all memory types"""
    id: str
    timestamp: datetime
    type: str
    content: Dict
    context: Optional[Dict] = None
    participants: Optional[Set[str]] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        """Convert memory to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.type,
            'content': json.dumps(self.content),
            'context': json.dumps(self.context) if self.context else None,
            'participants': list(self.participants) if self.participants else None,
            'embedding': self.embedding
        }

    def to_vector_document(self) -> VectorDocument:
        """Convert memory to vector document"""
        return VectorDocument(
            id=self.id,
            text=json.dumps(self.content),  # Main content as text
            metadata={
                'type': self.type,
                'timestamp': self.timestamp.isoformat(),
                'context': self.context,
                'participants': list(self.participants) if self.participants else None
            },
            embedding=self.embedding,
            timestamp=self.timestamp
        )

class MemorySystem:
    """Unified memory system for the bot"""
    
    def __init__(self, db: Any, persist_directory: str):
        """Initialize memory system"""
        self.db = db
        self.vector_store = VectorStore(persist_directory)
        self.working_memory: List[Memory] = []
        self.working_memory_size = 100
        self.personality_state = {
            'mood': 'neutral',
            'energy': 1.0,
            'focus': 1.0,
            'last_action_time': datetime.now(),
            'recent_interactions': [],
            'active_conversations': set(),
            'current_topics': set(),
            'engagement_stats': {
                'tweets': 0,
                'replies': 0,
                'likes': 0,
                'retweets': 0
            }
        }
        self.attention_weights = {}
        
    def add_memory(self, memory: Memory):
        """Add a new memory"""
        try:
            # Convert to vector document and add to vector store
            vector_doc = memory.to_vector_document()
            collection_name = self._get_collection_for_type(memory.type)
            self.vector_store.add_document(vector_doc, collection_name)
            
            # Add to working memory
            self.working_memory.append(memory)
            if len(self.working_memory) > self.working_memory_size:
                self.working_memory.pop(0)
            
            # Update personality state
            self._update_personality_state(memory)
            
            # Store in database
            self.db.insert_memory(memory.to_dict())
            
            # Update working memory weights
            self._update_working_memory(memory)
            
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")

    def _update_working_memory(self, memory: Memory):
        """Update working memory with new memory and recalculate attention weights"""
        try:
            # Calculate attention weights
            self.attention_weights = {}
            total_weight = 0.0
            
            for mem in self.working_memory:
                # Base weight starts with recency
                age = datetime.now() - mem.timestamp
                recency_weight = 1.0 / (1.0 + age.total_seconds() / 3600)  # Decay over hours
                
                # Add importance factor
                importance = self._calculate_memory_importance(mem)
                
                # Add relevance to current context
                relevance = self._calculate_context_relevance(mem)
                
                # Combine weights
                weight = recency_weight * importance * relevance
                self.attention_weights[mem.id] = weight
                total_weight += weight
            
            # Normalize weights with safeguard against zero division
            if total_weight > 0:
                for mem_id in self.attention_weights:
                    self.attention_weights[mem_id] /= total_weight
            else:
                # If all weights are zero, use uniform weighting
                uniform_weight = 1.0 / len(self.working_memory) if self.working_memory else 0.0
                for mem_id in self.attention_weights:
                    self.attention_weights[mem_id] = uniform_weight
            
        except Exception as e:
            logger.error(f"Error updating working memory: {str(e)}")

    def _calculate_memory_importance(self, memory: Memory) -> float:
        """Calculate importance score for a memory"""
        try:
            importance = 1.0
            
            # Check memory type
            if memory.type == 'interaction':
                content = memory.content
                if isinstance(content, str):
                    content = json.loads(content)
                
                # Consider engagement metrics
                engagement = content.get('engagement', {})
                likes = engagement.get('likes', 0)
                replies = engagement.get('replies', 0)
                importance *= (1.0 + (likes * 0.1 + replies * 0.2))
                
                # Consider sentiment intensity
                sentiment = abs(content.get('sentiment', 0))
                importance *= (1.0 + sentiment)
            
            elif memory.type in ['knowledge', 'pattern', 'summary']:
                # Knowledge and patterns are generally more important
                importance *= 1.5
                
                # Consider confidence if available
                confidence = memory.content.get('confidence', 0.5)
                importance *= (0.5 + confidence)
            
            # Cap importance to avoid extreme values
            return min(importance, 5.0)
            
        except Exception as e:
            logger.error(f"Error calculating memory importance: {str(e)}")
            return 1.0

    def _calculate_context_relevance(self, memory: Memory) -> float:
        """Calculate relevance to current context"""
        try:
            relevance = 1.0
            
            # Get current context
            current_topics = self.personality_state.get('current_topics', set())
            active_conversations = self.personality_state.get('active_conversations', set())
            
            if memory.context:
                # Check topic overlap
                memory_topics = set(memory.context.get('topics', []))
                topic_overlap = len(memory_topics & current_topics)
                if topic_overlap > 0:
                    relevance *= (1.0 + (topic_overlap * 0.2))
                
                # Check conversation relevance
                conv_id = memory.context.get('conversation_id')
                if conv_id and conv_id in active_conversations:
                    relevance *= 1.5
                
                # Check participant overlap
                if memory.participants:
                    recent_participants = {
                        p for m in self.working_memory[-5:]
                        for p in (m.participants or set())
                    }
                    participant_overlap = len(memory.participants & recent_participants)
                    if participant_overlap > 0:
                        relevance *= (1.0 + (participant_overlap * 0.1))
            
            # Cap relevance to avoid extreme values
            return min(relevance, 3.0)
            
        except Exception as e:
            logger.error(f"Error calculating context relevance: {str(e)}")
            return 1.0

    def get_recent_memories(self, 
                          memory_type: Optional[str] = None,
                          limit: int = 10) -> List[Memory]:
        """Get recent memories, optionally filtered by type"""
        try:
            query = "SELECT * FROM memories"
            if memory_type:
                query += f" WHERE type = '{memory_type}'"
            query += " ORDER BY timestamp DESC LIMIT ?"
            
            memories = []
            for row in self.db.execute(query, [limit]):
                memories.append(self._row_to_memory(row))
            return memories
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {str(e)}")
            return []
    
    def search_memories(
        self,
        query: str,
        search_type: str = 'semantic',
        filters: Optional[Dict] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Memory]:
        """
        Advanced memory search with multiple strategies.
        
        Search types:
        - semantic: Search by meaning similarity
        - temporal: Search within time ranges
        - causal: Search for cause-effect relationships
        - multi_hop: Search through memory chains
        - contextual: Search by context and relationships
        - combined: Combine multiple search strategies
        """
        try:
            if search_type == 'semantic':
                return self._semantic_search(query, filters, limit)
            elif search_type == 'temporal':
                return self._temporal_search(query, filters, limit, **kwargs)
            elif search_type == 'causal':
                return self._causal_search(query, filters, limit)
            elif search_type == 'multi_hop':
                return self._multi_hop_search(query, filters, limit, **kwargs)
            elif search_type == 'contextual':
                return self._contextual_search(query, filters, limit)
            elif search_type == 'combined':
                return self._combined_search(query, filters, limit, **kwargs)
            else:
                logger.warning(f"Unknown search type: {search_type}")
                return self._semantic_search(query, filters, limit)
                
        except Exception as e:
            logger.error(f"Error in search_memories: {str(e)}")
            return []

    def _semantic_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Semantic search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.vector_store.generate_embedding(query)
            
            # Prepare search filters
            search_filters = self._prepare_filters(filters)
            
            # Search across all relevant collections
            all_results = []
            for collection_name in [
                VectorStore.MEMORIES_COLLECTION,
                VectorStore.CONTENT_COLLECTION,
                VectorStore.KNOWLEDGE_COLLECTION
            ]:
                results = self.vector_store.query_similar(
                    query_embedding,
                    collection_name,
                    n_results=limit,
                    filters=search_filters.get(collection_name)
                )
                all_results.extend(self._vector_docs_to_memories(results))
            
            # Sort by relevance and return top results
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

    def _temporal_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 10,
        time_range: Optional[Dict] = None,
        temporal_order: str = 'chronological'
    ) -> List[Memory]:
        """Search memories with temporal constraints"""
        try:
            # Get base results
            base_results = self._semantic_search(query, filters, limit * 2)
            
            # Apply temporal filtering
            if time_range:
                start_time = datetime.fromisoformat(time_range.get('start', '2000-01-01'))
                end_time = datetime.fromisoformat(time_range.get('end', datetime.now().isoformat()))
                
                base_results = [
                    m for m in base_results
                    if start_time <= m.timestamp <= end_time
                ]
            
            # Sort results
            if temporal_order == 'chronological':
                base_results.sort(key=lambda x: x.timestamp)
            elif temporal_order == 'reverse':
                base_results.sort(key=lambda x: x.timestamp, reverse=True)
            
            return base_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in temporal search: {str(e)}")
            return []

    def _causal_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Search for cause-effect relationships"""
        try:
            # Get initial memory that matches query
            initial_results = self._semantic_search(query, filters, 1)
            if not initial_results:
                return []
            
            target_memory = initial_results[0]
            
            # Find related memories before (potential causes)
            before_query = {
                'end_time': target_memory.timestamp,
                'window': timedelta(hours=24)
            }
            potential_causes = self._temporal_search(
                query,
                filters,
                limit // 2,
                time_range=before_query,
                temporal_order='reverse'
            )
            
            # Find related memories after (potential effects)
            after_query = {
                'start_time': target_memory.timestamp,
                'window': timedelta(hours=24)
            }
            potential_effects = self._temporal_search(
                query,
                filters,
                limit // 2,
                time_range=after_query,
                temporal_order='chronological'
            )
            
            # Combine and order results
            results = []
            results.extend(potential_causes)
            results.append(target_memory)
            results.extend(potential_effects)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in causal search: {str(e)}")
            return []

    def _multi_hop_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 10,
        max_hops: int = 3
    ) -> List[Memory]:
        """Search through chains of related memories"""
        try:
            # Get initial results
            current_results = self._semantic_search(query, filters, limit)
            if not current_results:
                return []
            
            all_results = set(current_results)
            seen_ids = {m.id for m in current_results}
            
            # Perform multi-hop search
            for hop in range(max_hops):
                next_hop_results = []
                
                # For each current result, find related memories
                for memory in current_results:
                    # Search by shared context
                    context_results = self._contextual_search(
                        memory.content.get('text', ''),
                        {'context': memory.context},
                        limit // 2
                    )
                    
                    # Search by temporal proximity
                    temporal_results = self._temporal_search(
                        memory.content.get('text', ''),
                        None,
                        limit // 2,
                        {
                            'start': (memory.timestamp - timedelta(hours=1)).isoformat(),
                            'end': (memory.timestamp + timedelta(hours=1)).isoformat()
                        }
                    )
                    
                    # Add new results
                    for result in context_results + temporal_results:
                        if result.id not in seen_ids:
                            next_hop_results.append(result)
                            seen_ids.add(result.id)
                
                if not next_hop_results:
                    break
                
                current_results = next_hop_results
                all_results.update(next_hop_results)
            
            # Sort by relevance to original query
            results_list = list(all_results)
            results_list.sort(
                key=lambda x: self._calculate_relevance(x, query),
                reverse=True
            )
            
            return results_list[:limit]
            
        except Exception as e:
            logger.error(f"Error in multi-hop search: {str(e)}")
            return []

    def _contextual_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Search based on context and relationships"""
        try:
            # Get base results
            base_results = self._semantic_search(query, filters, limit * 2)
            if not base_results:
                return []
            
            # Extract contexts from base results
            contexts = set()
            for memory in base_results:
                if memory.context:
                    # Add conversation contexts
                    conv_id = memory.context.get('conversation_id')
                    if conv_id:
                        contexts.add(('conversation', conv_id))
                    
                    # Add participant contexts
                    if memory.participants:
                        for participant in memory.participants:
                            contexts.add(('participant', participant))
                    
                    # Add topic contexts
                    topics = memory.context.get('topics', [])
                    for topic in topics:
                        contexts.add(('topic', topic))
            
            # Search for memories sharing these contexts
            contextual_results = set()
            for context_type, context_value in contexts:
                context_filter = {context_type: context_value}
                results = self._semantic_search(query, context_filter, limit)
                contextual_results.update(results)
            
            # Sort by relevance and context overlap
            results_list = list(contextual_results)
            results_list.sort(
                key=lambda x: (
                    self._calculate_relevance(x, query) +
                    self._calculate_context_overlap(x, contexts)
                ),
                reverse=True
            )
            
            return results_list[:limit]
            
        except Exception as e:
            logger.error(f"Error in contextual search: {str(e)}")
            return []

    def _combined_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 10,
        strategies: Optional[List[str]] = None
    ) -> List[Memory]:
        """Combine multiple search strategies"""
        try:
            if not strategies:
                strategies = ['semantic', 'temporal', 'contextual']
            
            all_results = set()
            
            # Execute each search strategy
            for strategy in strategies:
                if strategy == 'semantic':
                    results = self._semantic_search(query, filters, limit)
                elif strategy == 'temporal':
                    results = self._temporal_search(query, filters, limit)
                elif strategy == 'contextual':
                    results = self._contextual_search(query, filters, limit)
                elif strategy == 'causal':
                    results = self._causal_search(query, filters, limit)
                elif strategy == 'multi_hop':
                    results = self._multi_hop_search(query, filters, limit)
                else:
                    continue
                
                all_results.update(results)
            
            # Rank combined results
            results_list = list(all_results)
            results_list.sort(
                key=lambda x: self._calculate_combined_score(x, query, strategies),
                reverse=True
            )
            
            return results_list[:limit]
            
        except Exception as e:
            logger.error(f"Error in combined search: {str(e)}")
            return []

    def _calculate_relevance(self, memory: Memory, query: str) -> float:
        """Calculate relevance score between memory and query"""
        try:
            # Generate embeddings
            query_embedding = self.vector_store.generate_embedding(query)
            memory_embedding = memory.embedding
            
            if not memory_embedding:
                memory_text = json.dumps(memory.content)
                memory_embedding = self.vector_store.generate_embedding(memory_text)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, memory_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            return 0.0

    def _calculate_context_overlap(
        self,
        memory: Memory,
        contexts: Set[tuple]
    ) -> float:
        """Calculate context overlap score"""
        try:
            if not memory.context:
                return 0.0
            
            overlap_count = 0
            total_contexts = len(contexts)
            
            for context_type, context_value in contexts:
                if context_type == 'conversation':
                    if memory.context.get('conversation_id') == context_value:
                        overlap_count += 1
                
                elif context_type == 'participant':
                    if memory.participants and context_value in memory.participants:
                        overlap_count += 1
                
                elif context_type == 'topic':
                    if context_value in memory.context.get('topics', []):
                        overlap_count += 1
            
            return overlap_count / total_contexts if total_contexts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating context overlap: {str(e)}")
            return 0.0

    def _calculate_combined_score(
        self,
        memory: Memory,
        query: str,
        strategies: List[str]
    ) -> float:
        """Calculate combined ranking score"""
        try:
            scores = []
            
            # Calculate semantic similarity
            if 'semantic' in strategies:
                scores.append(self._calculate_relevance(memory, query))
            
            # Calculate temporal relevance
            if 'temporal' in strategies:
                age = datetime.now() - memory.timestamp
                temporal_score = 1.0 / (1.0 + age.total_seconds() / 86400)  # Decay over days
                scores.append(temporal_score)
            
            # Calculate contextual relevance
            if 'contextual' in strategies:
                base_results = self._semantic_search(query, None, 5)
                contexts = set()
                for result in base_results:
                    if result.context:
                        for k, v in result.context.items():
                            if isinstance(v, (str, int, float)):
                                contexts.add((k, str(v)))
                context_score = self._calculate_context_overlap(memory, contexts)
                scores.append(context_score)
            
            # Return weighted average
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating combined score: {str(e)}")
            return 0.0

    def get_personality_state(self) -> Dict:
        """Get current personality state"""
        return self.personality_state.copy()
    
    def update_personality_state(self, updates: Dict):
        """Update personality state with new values"""
        try:
            self.personality_state.update(updates)
            
            # Store state change in memory
            self.add_memory(Memory(
                id=f"state_change_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                type='personality_state_change',
                content=updates,
                context={'previous_state': self.personality_state}
            ))
            
        except Exception as e:
            logger.error(f"Error updating personality state: {str(e)}")
    
    def get_conversation_context(self, 
                               conversation_id: str,
                               limit: int = 10) -> List[Memory]:
        """Get context for a conversation"""
        try:
            query = """
                SELECT * FROM memories 
                WHERE type IN ('tweet', 'reply', 'mention') 
                AND json_extract(content, '$.conversation_id') = ?
                ORDER BY timestamp DESC LIMIT ?
            """
            
            memories = []
            for row in self.db.execute(query, [conversation_id, limit]):
                memories.append(self._row_to_memory(row))
            return memories
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return []
    
    def get_user_interaction_history(self,
                                   user_id: str,
                                   limit: int = 20) -> List[Memory]:
        """Get history of interactions with a user"""
        try:
            query = """
                SELECT * FROM memories 
                WHERE json_array_contains(participants, ?) = 1
                ORDER BY timestamp DESC LIMIT ?
            """
            
            memories = []
            for row in self.db.execute(query, [user_id, limit]):
                memories.append(self._row_to_memory(row))
            return memories
            
        except Exception as e:
            logger.error(f"Error getting user history: {str(e)}")
            return []
    
    def consolidate_memories(self, force: bool = False):
        """
        Consolidate memories into long-term storage and extract patterns.
        This process includes:
        1. Summarizing recent interactions
        2. Extracting recurring patterns
        3. Updating knowledge base
        4. Pruning old memories
        5. Optimizing vector store
        """
        try:
            current_time = datetime.now()
            last_consolidation = self.personality_state.get('last_consolidation', 
                datetime.now() - timedelta(hours=25))
            
            # Check if consolidation is needed
            if not force and (current_time - last_consolidation) < timedelta(hours=24):
                logger.info("Skipping consolidation - too recent")
                return
            
            logger.info("Starting memory consolidation")
            
            # Get recent memories to consolidate
            consolidation_window = timedelta(days=7)
            recent_memories = self.get_recent_memories(limit=1000)
            consolidation_memories = [
                m for m in recent_memories 
                if current_time - m.timestamp < consolidation_window
            ]
            
            if not consolidation_memories:
                logger.info("No memories to consolidate")
                return
            
            # Group memories by type and context
            grouped_memories = self._group_memories_for_consolidation(consolidation_memories)
            
            # Process each group
            for group_key, memories in grouped_memories.items():
                # Generate summaries
                summary = self._generate_consolidated_summary(memories)
                if summary:
                    summary_memory = Memory(
                        id=f"summary_{current_time.timestamp()}_{group_key}",
                        timestamp=current_time,
                        type='summary',
                        content={
                            'summary': summary,
                            'source_count': len(memories),
                            'time_range': {
                                'start': min(m.timestamp for m in memories).isoformat(),
                                'end': max(m.timestamp for m in memories).isoformat()
                            },
                            'group_key': group_key
                        },
                        context={'consolidated_from': [m.id for m in memories]}
                    )
                    self.add_memory(summary_memory)
                
                # Extract patterns
                patterns = self._extract_patterns(memories)
                if patterns:
                    pattern_memory = Memory(
                        id=f"pattern_{current_time.timestamp()}_{group_key}",
                        timestamp=current_time,
                        type='pattern',
                        content={
                            'patterns': patterns,
                            'source_count': len(memories),
                            'group_key': group_key
                        },
                        context={'consolidated_from': [m.id for m in memories]}
                    )
                    self.add_memory(pattern_memory)
                
                # Extract knowledge
                knowledge = self._extract_knowledge(memories)
                if knowledge:
                    knowledge_memory = Memory(
                        id=f"knowledge_{current_time.timestamp()}_{group_key}",
                        timestamp=current_time,
                        type='knowledge',
                        content={
                            'knowledge': knowledge,
                            'confidence': knowledge.get('confidence', 0.0),
                            'source_count': len(memories)
                        },
                        context={'consolidated_from': [m.id for m in memories]}
                    )
                    self.add_memory(knowledge_memory)
            
            # Prune old memories
            self._prune_old_memories()
            
            # Optimize vector store
            self.vector_store.clear_cache()
            
            # Update consolidation timestamp
            self.update_personality_state({
                'last_consolidation': current_time
            })
            
            logger.info("Memory consolidation complete")
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {str(e)}")

    def _group_memories_for_consolidation(self, memories: List[Memory]) -> Dict[str, List[Memory]]:
        """Group memories by type and context for consolidation"""
        groups = {}
        
        for memory in memories:
            # Create group key based on type and relevant context
            group_components = [memory.type]
            
            if memory.type == 'interaction':
                # Group by conversation or participant
                conv_id = memory.context.get('conversation_id') if memory.context else None
                if conv_id:
                    group_components.append(f"conv_{conv_id}")
                elif memory.participants:
                    group_components.append(f"participants_{'_'.join(sorted(memory.participants))}")
            
            elif memory.type == 'knowledge':
                # Group by topic or domain
                topic = memory.context.get('topic') if memory.context else None
                if topic:
                    group_components.append(f"topic_{topic}")
            
            group_key = '_'.join(group_components)
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(memory)
        
        return groups

    def _generate_consolidated_summary(self, memories: List[Memory]) -> Optional[str]:
        """Generate a consolidated summary of related memories"""
        try:
            if not memories:
                return None
            
            # Sort by timestamp
            memories = sorted(memories, key=lambda m: m.timestamp)
            
            # Extract key information
            key_points = []
            for memory in memories:
                if memory.type == 'interaction':
                    content = memory.content
                    if isinstance(content, str):
                        content = json.loads(content)
                    key_points.append({
                        'timestamp': memory.timestamp,
                        'type': 'interaction',
                        'content': content.get('text', ''),
                        'sentiment': content.get('sentiment', 0)
                    })
                elif memory.type == 'knowledge':
                    content = memory.content
                    if isinstance(content, str):
                        content = json.loads(content)
                    key_points.append({
                        'timestamp': memory.timestamp,
                        'type': 'knowledge',
                        'content': content.get('fact', ''),
                        'confidence': content.get('confidence', 0)
                    })
            
            if not key_points:
                return None
            
            # Generate summary based on memory type
            if memories[0].type == 'interaction':
                return self._summarize_interactions(key_points)
            elif memories[0].type == 'knowledge':
                return self._summarize_knowledge(key_points)
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating consolidated summary: {str(e)}")
            return None

    def _extract_patterns(self, memories: List[Memory]) -> Optional[Dict]:
        """Extract recurring patterns from memories"""
        try:
            if not memories or len(memories) < 3:  # Need at least 3 memories for pattern
                return None
            
            patterns = {
                'temporal': self._analyze_temporal_patterns(memories),
                'sentiment': self._analyze_sentiment_patterns(memories),
                'engagement': self._analyze_engagement_patterns(memories),
                'topics': self._analyze_topic_patterns(memories)
            }
            
            # Filter out empty patterns
            patterns = {k: v for k, v in patterns.items() if v}
            
            return patterns if patterns else None
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
            return None

    def _extract_knowledge(self, memories: List[Memory]) -> Optional[Dict]:
        """Extract knowledge from memories"""
        try:
            if not memories:
                return None
            
            # Extract facts and assertions
            facts = []
            total_confidence = 0.0
            
            for memory in memories:
                content = memory.content
                if isinstance(content, str):
                    content = json.loads(content)
                
                if memory.type == 'knowledge':
                    fact = content.get('fact')
                    confidence = content.get('confidence', 0.5)
                    if fact:
                        facts.append({
                            'fact': fact,
                            'confidence': confidence,
                            'timestamp': memory.timestamp.isoformat()
                        })
                        total_confidence += confidence
                
                elif memory.type == 'interaction':
                    # Extract potential knowledge from interactions
                    text = content.get('text', '')
                    if text and any(marker in text.lower() for marker in [
                        'always', 'never', 'every', 'all', 'none', 'must', 'should'
                    ]):
                        facts.append({
                            'fact': text,
                            'confidence': 0.3,  # Lower confidence for extracted facts
                            'timestamp': memory.timestamp.isoformat()
                        })
                        total_confidence += 0.3
            
            if not facts:
                return None
            
            # Aggregate knowledge
            return {
                'facts': facts,
                'confidence': total_confidence / len(facts),
                'source_count': len(memories)
            }
            
        except Exception as e:
            logger.error(f"Error extracting knowledge: {str(e)}")
            return None

    def _prune_old_memories(self):
        """Prune old memories based on relevance and age"""
        try:
            current_time = datetime.now()
            
            # Get all memories older than 30 days
            old_memories = self.get_recent_memories(limit=10000)
            old_memories = [
                m for m in old_memories 
                if (current_time - m.timestamp) > timedelta(days=30)
            ]
            
            if not old_memories:
                return
            
            # Keep memories that are:
            # 1. Part of active knowledge
            # 2. Referenced recently
            # 3. Have high engagement
            # 4. Are summaries or patterns
            keep_ids = set()
            
            for memory in old_memories:
                should_keep = False
                
                # Check memory type
                if memory.type in ['summary', 'pattern', 'knowledge']:
                    should_keep = True
                
                # Check references
                elif memory.id in [
                    ref for m in self.get_recent_memories(limit=100)
                    for ref in m.context.get('references', [])
                ]:
                    should_keep = True
                
                # Check engagement
                elif memory.type == 'interaction':
                    content = memory.content
                    if isinstance(content, str):
                        content = json.loads(content)
                    engagement = content.get('engagement', {})
                    if engagement.get('likes', 0) > 10 or engagement.get('replies', 0) > 5:
                        should_keep = True
                
                if should_keep:
                    keep_ids.add(memory.id)
            
            # Remove memories not in keep_ids
            for memory in old_memories:
                if memory.id not in keep_ids:
                    # Remove from vector store
                    collection_name = self._get_collection_for_type(memory.type)
                    self.vector_store.delete_document(memory.id, collection_name)
                    
                    # Remove from database
                    self.db.delete_memory(memory.id)
            
        except Exception as e:
            logger.error(f"Error pruning old memories: {str(e)}")

    def _update_personality_state(self, memory: Memory):
        """Update personality state based on new memory"""
        try:
            # Update last action time
            self.personality_state['last_action_time'] = memory.timestamp
            
            # Update engagement stats
            if memory.type in ['tweet', 'reply', 'retweet', 'like']:
                self.personality_state['engagement_stats'][f"{memory.type}s"] += 1
            
            # Update active conversations
            if memory.type in ['tweet', 'reply', 'mention']:
                conversation_id = memory.content.get('conversation_id')
                if conversation_id:
                    self.personality_state['active_conversations'].add(conversation_id)
            
            # Update current topics
            if 'topics' in memory.content:
                self.personality_state['current_topics'].update(
                    memory.content['topics']
                )
            
            # Update recent interactions
            if memory.type in ['tweet', 'reply', 'mention', 'like', 'retweet']:
                self.personality_state['recent_interactions'].append({
                    'type': memory.type,
                    'timestamp': memory.timestamp.isoformat(),
                    'content_summary': str(memory.content)[:100]
                })
                # Keep only recent interactions
                if len(self.personality_state['recent_interactions']) > 20:
                    self.personality_state['recent_interactions'].pop(0)
            
            # Update mood and energy based on interaction type
            if memory.type in ['reply', 'mention']:
                # Increase energy for interactions
                self.personality_state['energy'] = min(
                    1.0,
                    self.personality_state['energy'] + 0.1
                )
            elif memory.type == 'research':
                # Increase focus for research activities
                self.personality_state['focus'] = min(
                    1.0,
                    self.personality_state['focus'] + 0.1
                )
            
        except Exception as e:
            logger.error(f"Error updating personality state: {str(e)}")
    
    def _row_to_memory(self, row: Dict) -> Memory:
        """Convert database row to Memory object"""
        return Memory(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            type=row['type'],
            content=json.loads(row['content']),
            context=json.loads(row['context']) if row['context'] else None,
            participants=set(json.loads(row['participants'])) if row['participants'] else None,
            embedding=row['embedding']
        )
    
    def _generate_memory_summary(self, memories: List[Memory]) -> str:
        """Generate a summary of a group of memories"""
        try:
            # Implement memory summarization logic here
            # This could use an LLM to generate a natural language summary
            # For now, just return basic stats
            return f"Group of {len(memories)} memories of type {memories[0].type}"
        except Exception as e:
            logger.error(f"Error generating memory summary: {str(e)}")
            return "Error generating summary"

    def _get_collection_for_type(self, memory_type: Optional[str]) -> str:
        """Get the appropriate collection name for a memory type"""
        if not memory_type:
            return VectorStore.MEMORIES_COLLECTION
            
        type_mapping = {
            'interaction': VectorStore.CONTENT_COLLECTION,
            'knowledge': VectorStore.KNOWLEDGE_COLLECTION,
            'personality': VectorStore.MEMORIES_COLLECTION,
            'system': VectorStore.MEMORIES_COLLECTION
        }
        
        return type_mapping.get(memory_type, VectorStore.MEMORIES_COLLECTION)

    def _prepare_filters(self, filters: Optional[Dict]) -> Dict:
        """Prepare filters for search"""
        if not filters:
            return {}
        
        prepared_filters = {}
        
        for key, value in filters.items():
            if isinstance(value, str):
                prepared_filters[key] = value
            elif isinstance(value, list):
                prepared_filters[key] = ','.join(value)
            else:
                prepared_filters[key] = str(value)
        
        return prepared_filters

    def _vector_docs_to_memories(self, vector_docs: List[VectorDocument]) -> List[Memory]:
        """Convert vector documents to memories"""
        memories = []
        
        for doc in vector_docs:
            memory = Memory(
                id=doc.id,
                timestamp=doc.timestamp or datetime.now(),
                type=doc.metadata['type'],
                content=json.loads(doc.text),
                context=doc.metadata.get('context'),
                participants=set(doc.metadata.get('participants', [])) if doc.metadata.get('participants') else None,
                embedding=doc.embedding
            )
            memories.append(memory)
        
        return memories
