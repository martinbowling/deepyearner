"""
Unified vector store for managing all embeddings in the system.
Handles both episodic memories and external content in a single interface.
"""
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import json
import os
from dataclasses import dataclass, asdict
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
import tiktoken
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class VectorDocument:
    """Represents a document in the vector store"""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        d = asdict(self)
        if self.timestamp:
            d['timestamp'] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorDocument':
        """Create from dictionary format"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class VectorStore:
    """Unified vector store for all embeddings"""
    
    # Collection names
    MEMORIES_COLLECTION = "memories"
    CONTENT_COLLECTION = "content"
    KNOWLEDGE_COLLECTION = "knowledge"
    
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384
    ):
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )
        
        # Initialize collections
        self._init_collections()
        
        # Initialize tokenizer for text chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Cache for recently accessed documents
        self.document_cache: Dict[str, VectorDocument] = {}
    
    def _init_collections(self):
        """Initialize ChromaDB collections"""
        try:
            # Memories collection (episodic memories)
            self.memories_collection = self.chroma_client.get_or_create_collection(
                name=self.MEMORIES_COLLECTION,
                metadata={"description": "Episodic memories"}
            )
            
            # Content collection (tweets, replies, etc)
            self.content_collection = self.chroma_client.get_or_create_collection(
                name=self.CONTENT_COLLECTION,
                metadata={"description": "Social media content"}
            )
            
            # Knowledge collection (learned information)
            self.knowledge_collection = self.chroma_client.get_or_create_collection(
                name=self.KNOWLEDGE_COLLECTION,
                metadata={"description": "Learned knowledge"}
            )
            
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            raise
    
    def add_document(
        self,
        document: VectorDocument,
        collection_name: str
    ) -> bool:
        """Add a document to the specified collection"""
        try:
            # Get the appropriate collection
            collection = self._get_collection(collection_name)
            if not collection:
                return False
            
            # Generate embedding if not provided
            if document.embedding is None:
                document.embedding = self.generate_embedding(document.text)
            
            # Add to ChromaDB
            collection.add(
                ids=[document.id],
                embeddings=[document.embedding],
                metadatas=[document.metadata],
                documents=[document.text]
            )
            
            # Add to cache
            self.document_cache[document.id] = document
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False
    
    def add_documents(
        self,
        documents: List[VectorDocument],
        collection_name: str
    ) -> bool:
        """Add multiple documents to the specified collection"""
        try:
            # Get the appropriate collection
            collection = self._get_collection(collection_name)
            if not collection:
                return False
            
            # Prepare batch data
            ids = []
            embeddings = []
            metadatas = []
            texts = []
            
            for doc in documents:
                # Generate embedding if not provided
                if doc.embedding is None:
                    doc.embedding = self.generate_embedding(doc.text)
                
                ids.append(doc.id)
                embeddings.append(doc.embedding)
                metadatas.append(doc.metadata)
                texts.append(doc.text)
                
                # Add to cache
                self.document_cache[doc.id] = doc
            
            # Add to ChromaDB in batch
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents in batch: {str(e)}")
            return False
    
    def query_similar(
        self,
        query: Union[str, List[float]],
        collection_name: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[VectorDocument]:
        """Query for similar documents"""
        try:
            # Get the appropriate collection
            collection = self._get_collection(collection_name)
            if not collection:
                return []
            
            # Generate embedding if query is text
            if isinstance(query, str):
                query_embedding = self.generate_embedding(query)
            else:
                query_embedding = query
            
            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            # Convert results to VectorDocuments
            documents = []
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                
                # Check similarity threshold
                if results['distances'][0][i] > min_similarity:
                    continue
                
                # Get from cache or create new
                if doc_id in self.document_cache:
                    documents.append(self.document_cache[doc_id])
                else:
                    doc = VectorDocument(
                        id=doc_id,
                        text=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        embedding=results['embeddings'][0][i] if 'embeddings' in results else None
                    )
                    self.document_cache[doc_id] = doc
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error querying similar documents: {str(e)}")
            return []
    
    def update_document(
        self,
        document: VectorDocument,
        collection_name: str
    ) -> bool:
        """Update an existing document"""
        try:
            # Get the appropriate collection
            collection = self._get_collection(collection_name)
            if not collection:
                return False
            
            # Generate new embedding if needed
            if document.embedding is None:
                document.embedding = self.generate_embedding(document.text)
            
            # Update in ChromaDB
            collection.update(
                ids=[document.id],
                embeddings=[document.embedding],
                metadatas=[document.metadata],
                documents=[document.text]
            )
            
            # Update cache
            self.document_cache[document.id] = document
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False
    
    def delete_document(
        self,
        document_id: str,
        collection_name: str
    ) -> bool:
        """Delete a document"""
        try:
            # Get the appropriate collection
            collection = self._get_collection(collection_name)
            if not collection:
                return False
            
            # Delete from ChromaDB
            collection.delete(ids=[document_id])
            
            # Remove from cache
            if document_id in self.document_cache:
                del self.document_cache[document_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            # Truncate text if too long
            max_tokens = 512
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                text = self.tokenizer.decode(tokens[:max_tokens])
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * self.embedding_dimension
    
    def _get_collection(self, name: str) -> Optional[Collection]:
        """Get ChromaDB collection by name"""
        if name == self.MEMORIES_COLLECTION:
            return self.memories_collection
        elif name == self.CONTENT_COLLECTION:
            return self.content_collection
        elif name == self.KNOWLEDGE_COLLECTION:
            return self.knowledge_collection
        else:
            logger.error(f"Unknown collection name: {name}")
            return None
    
    def get_document(
        self,
        document_id: str,
        collection_name: str
    ) -> Optional[VectorDocument]:
        """Get a document by ID"""
        try:
            # Check cache first
            if document_id in self.document_cache:
                return self.document_cache[document_id]
            
            # Get the appropriate collection
            collection = self._get_collection(collection_name)
            if not collection:
                return None
            
            # Get from ChromaDB
            result = collection.get(
                ids=[document_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            if not result['ids']:
                return None
            
            # Create VectorDocument
            doc = VectorDocument(
                id=document_id,
                text=result['documents'][0],
                metadata=result['metadatas'][0],
                embedding=result['embeddings'][0] if 'embeddings' in result else None
            )
            
            # Add to cache
            self.document_cache[document_id] = doc
            
            return doc
            
        except Exception as e:
            logger.error(f"Error getting document: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear the document cache"""
        self.document_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            stats = {
                "total_documents": {
                    "memories": self.memories_collection.count(),
                    "content": self.content_collection.count(),
                    "knowledge": self.knowledge_collection.count()
                },
                "cache_size": len(self.document_cache),
                "embedding_dimension": self.embedding_dimension,
                "persist_directory": self.persist_directory
            }
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
