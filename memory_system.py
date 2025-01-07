"""
Unified memory system that handles all types of memory including personality state.
Provides a centralized interface for storing and retrieving memories, experiences,
and state information.
"""
import logging
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict, field
import numpy as np
import sqlite3
import uuid

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """Represents a memory that can be stored and retrieved"""
    id: str
    type: str
    content: str
    timestamp: str
    context: Union[str, Dict] = field(default_factory=dict)
    source: str = 'unknown'

    def to_vector_metadata(self) -> Dict:
        """Convert memory to vector store metadata"""
        return {
            'id': self.id,
            'type': self.type,
            'timestamp': self.timestamp,
            'context': json.dumps(self.context) if isinstance(self.context, dict) else self.context,
            'source': self.source
        }

class MemorySystem:
    """Manages storage and retrieval of memories"""
    
    def __init__(self, db: sqlite3.Connection, persist_directory: str):
        """Initialize memory system with database connection"""
        self.db = db
        self.persist_directory = persist_directory

    async def add_memory(self, memory_data: Union[Memory, Dict]) -> None:
        """Add a memory to both vector store and SQLite"""
        try:
            # Convert dict to Memory object if needed
            if isinstance(memory_data, dict):
                memory = Memory(
                    id=str(uuid.uuid4()),
                    type=memory_data['type'],
                    content=memory_data['content'],
                    timestamp=memory_data['timestamp'],
                    context=memory_data.get('context', '{}'),
                    source=memory_data.get('source', 'unknown')
                )
            else:
                memory = memory_data

            # Add to SQLite
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO memories 
                (id, type, content, timestamp, context, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.type,
                memory.content,
                memory.timestamp,
                json.dumps(memory.context) if isinstance(memory.context, dict) else memory.context,
                memory.source
            ))
            self.db.commit()

        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")
            raise

    async def get_recent_memories(self, hours: int = 24, limit: int = 100) -> List[Memory]:
        """Get recent memories from the last N hours"""
        try:
            cursor = self.db.cursor()
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute("""
                SELECT id, type, content, timestamp, context, source
                FROM memories
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (cutoff, limit))
            
            rows = cursor.fetchall()
            memories = []
            
            for row in rows:
                memories.append(Memory(
                    id=row[0],
                    type=row[1],
                    content=row[2],
                    timestamp=row[3],
                    context=json.loads(row[4]) if row[4] else {},
                    source=row[5]
                ))
                
            return memories
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {str(e)}")
            return []

    async def search_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories using SQLite full-text search"""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT id, type, content, timestamp, context, source
                FROM memories
                WHERE content MATCH ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (query, limit))
            
            rows = cursor.fetchall()
            memories = []
            
            for row in rows:
                memories.append(Memory(
                    id=row[0],
                    type=row[1],
                    content=row[2],
                    timestamp=row[3],
                    context=json.loads(row[4]) if row[4] else {},
                    source=row[5]
                ))
                
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return []
