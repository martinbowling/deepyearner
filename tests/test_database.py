"""
Tests for the database manager.
"""
import pytest
import sqlite3
from database import DatabaseManager, DatabaseError, ConnectionPool

def test_connection_pool_initialization(temp_db_path: str):
    """Test that the connection pool initializes correctly"""
    pool = ConnectionPool(max_connections=3)
    assert pool.max_connections == 3
    assert pool.connections.qsize() == 3
    pool.close_all()

def test_connection_pool_get_connection(temp_db_path: str):
    """Test getting a connection from the pool"""
    pool = ConnectionPool(max_connections=1)
    conn = pool.get_connection()
    assert isinstance(conn, sqlite3.Connection)
    assert pool.connections.qsize() == 0
    
    # Test timeout when no connections available
    assert pool.get_connection(timeout=0.1) is None
    
    # Return connection and verify it's back in pool
    pool.return_connection(conn)
    assert pool.connections.qsize() == 1
    pool.close_all()

def test_database_manager_initialization(temp_db_path: str):
    """Test database manager initialization"""
    db = DatabaseManager()
    
    # Verify tables were created
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Check oauth_tokens table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='oauth_tokens'
        """)
        assert cursor.fetchone() is not None
        
        # Check user_analysis table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='user_analysis'
        """)
        assert cursor.fetchone() is not None
        
        # Check analysis_queue table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='analysis_queue'
        """)
        assert cursor.fetchone() is not None
    
    db.close()

def test_database_manager_retry_logic(temp_db_path: str):
    """Test retry logic in database manager"""
    db = DatabaseManager()
    
    # Test successful query
    result = db.execute_with_retry(
        "SELECT 1 as test",
        retries=1
    )
    assert len(result) == 1
    assert dict(result[0])["test"] == 1
    
    # Test failed query with retry
    with pytest.raises(DatabaseError):
        db.execute_with_retry(
            "SELECT * FROM nonexistent_table",
            retries=2
        )
    
    db.close()

def test_database_manager_context_manager(temp_db_path: str):
    """Test database manager context manager"""
    db = DatabaseManager()
    
    # Test successful transaction
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        cursor.execute("INSERT INTO test (id) VALUES (1)")
    
    # Verify data was committed
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM test")
        assert cursor.fetchone() is not None
    
    # Test failed transaction
    with pytest.raises(DatabaseError):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO nonexistent_table VALUES (1)")
    
    db.close()

def test_database_manager_concurrent_access(temp_db_path: str):
    """Test concurrent access to database"""
    db = DatabaseManager()
    
    # Create test table
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
    
    # Simulate concurrent access
    def insert_data(value: int) -> None:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO test (id) VALUES (?)", (value,))
    
    # Insert data concurrently
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(insert_data, i) for i in range(5)]
        for future in futures:
            future.result()
    
    # Verify all data was inserted
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM test")
        assert dict(cursor.fetchone())["count"] == 5
    
    db.close() 