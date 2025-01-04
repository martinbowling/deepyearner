"""
Centralized database management for DeepYearner.
Provides connection pooling, retry logic, and proper error handling.
"""
import sqlite3
import threading
import time
from typing import Any, List, Dict, Optional
from contextlib import contextmanager
from queue import Queue, Empty
from config import config
from logger import get_logger

logger = get_logger(__name__)

class DatabaseError(Exception):
    """Base class for database errors"""
    pass

class ConnectionPool:
    """SQLite connection pool implementation"""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.connections: Queue[sqlite3.Connection] = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self._fill_pool()
    
    def _fill_pool(self) -> None:
        """Fill the connection pool"""
        for _ in range(self.max_connections):
            conn = sqlite3.connect(
                config.database.path,
                timeout=config.database.timeout,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            self.connections.put(conn)
    
    def get_connection(self, timeout: float = 5.0) -> Optional[sqlite3.Connection]:
        """Get a connection from the pool"""
        try:
            return self.connections.get(timeout=timeout)
        except Empty:
            logger.error("Timeout waiting for database connection")
            return None
    
    def return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool"""
        self.connections.put(conn)
    
    def close_all(self) -> None:
        """Close all connections in the pool"""
        while not self.connections.empty():
            conn = self.connections.get()
            conn.close()

class DatabaseManager:
    """Central database management class"""
    
    def __init__(self):
        """Initialize the database manager"""
        self.pool = ConnectionPool(max_connections=config.database.max_connections)
        self._ensure_tables()
    
    def _ensure_tables(self) -> None:
        """Ensure all required tables exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create oauth_tokens table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT,
                    token_type TEXT,
                    expires_at INTEGER,
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Create user_analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    engagement_score REAL,
                    topic_score REAL,
                    interaction_score REAL,
                    overall_score REAL,
                    analyzed_at INTEGER DEFAULT (strftime('%s', 'now')),
                    UNIQUE(user_id)
                )
            """)
            
            # Create analysis_queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    notes TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    processed_at INTEGER,
                    UNIQUE(username)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool"""
        conn = None
        try:
            conn = self.pool.get_connection()
            if conn is None:
                raise DatabaseError("Could not get database connection")
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}", exc_info=True)
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            if conn:
                self.pool.return_connection(conn)
    
    def execute_with_retry(self, query: str, params: tuple = (), retries: int = 3) -> Any:
        """Execute a query with retry logic"""
        last_error = None
        for attempt in range(retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    return cursor.fetchall()
            except DatabaseError as e:
                last_error = e
                if attempt < retries - 1:
                    delay = config.database.retry_delay * (attempt + 1)
                    logger.warning(f"Retrying database operation in {delay} seconds")
                    time.sleep(delay)
                continue
        raise last_error if last_error else DatabaseError("All retry attempts failed")
    
    def close(self) -> None:
        """Close the database manager and all connections"""
        self.pool.close_all()

# Global database instance
db = DatabaseManager() 