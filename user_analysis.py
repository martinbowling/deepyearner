"""
User analysis system for evaluating Twitter users as potential follows.
Tracks analysis history and provides utilities for managing target users/lists.
"""
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AnalysisSource(Enum):
    """Source of the user for analysis"""
    MANUAL = "manual"
    LIST = "list"
    RECOMMENDATION = "recommendation"

@dataclass
class UserAnalysis:
    """Results of analyzing a user"""
    user_id: str
    username: str
    analyzed_at: datetime
    engagement_rate: float
    topic_alignment: float
    interaction_quality: float
    overall_score: float
    recommendation: str
    source: AnalysisSource
    notes: str

class UserAnalysisSystem:
    """System for analyzing and tracking Twitter users"""
    
    def __init__(self, db_path: str = "bot.db"):
        """Initialize with database path"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create table for tracking analyzed users
        c.execute('''
            CREATE TABLE IF NOT EXISTS analyzed_users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                last_analyzed TEXT,
                engagement_rate REAL,
                topic_alignment REAL,
                interaction_quality REAL,
                overall_score REAL,
                recommendation TEXT,
                source TEXT,
                notes TEXT
            )
        ''')
        
        # Create table for tracking target users to analyze
        c.execute('''
            CREATE TABLE IF NOT EXISTS target_users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                added_at TEXT,
                source TEXT,
                source_list_id TEXT,
                status TEXT,
                priority INTEGER
            )
        ''')
        
        # Create table for tracking Twitter lists to monitor
        c.execute('''
            CREATE TABLE IF NOT EXISTS target_lists (
                list_id TEXT PRIMARY KEY,
                name TEXT,
                owner_id TEXT,
                added_at TEXT,
                last_checked TEXT,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_target_user(self, user_id: str, username: str, source: AnalysisSource, 
                       source_list_id: Optional[str] = None, priority: int = 1):
        """Add a user to the analysis target list"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if user was recently analyzed
            c.execute('''
                SELECT last_analyzed FROM analyzed_users 
                WHERE user_id = ? AND last_analyzed > ?
            ''', (
                user_id, 
                (datetime.now() - timedelta(days=30)).isoformat()
            ))
            
            recent = c.fetchone()
            if recent:
                logger.info(f"User {username} was recently analyzed on {recent[0]}")
                return False
            
            # Add to target list if not already present
            c.execute('''
                INSERT OR REPLACE INTO target_users 
                (user_id, username, added_at, source, source_list_id, status, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                username,
                datetime.now().isoformat(),
                source.value,
                source_list_id,
                'pending',
                priority
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding target user: {str(e)}")
            return False
    
    def add_target_list(self, list_id: str, name: str, owner_id: str):
        """Add a Twitter list to monitor for potential users"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                INSERT OR REPLACE INTO target_lists
                (list_id, name, owner_id, added_at, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                list_id,
                name,
                owner_id,
                datetime.now().isoformat(),
                'active'
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding target list: {str(e)}")
            return False
    
    def get_pending_users(self, limit: int = 10) -> List[Dict]:
        """Get users pending analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                SELECT user_id, username, source, source_list_id, priority
                FROM target_users
                WHERE status = 'pending'
                ORDER BY priority DESC, added_at ASC
                LIMIT ?
            ''', (limit,))
            
            users = [{
                'user_id': row[0],
                'username': row[1],
                'source': row[2],
                'source_list_id': row[3],
                'priority': row[4]
            } for row in c.fetchall()]
            
            conn.close()
            return users
            
        except Exception as e:
            logger.error(f"Error getting pending users: {str(e)}")
            return []
    
    def save_analysis(self, analysis: UserAnalysis):
        """Save analysis results for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Save analysis results
            c.execute('''
                INSERT OR REPLACE INTO analyzed_users
                (user_id, username, last_analyzed, engagement_rate, topic_alignment,
                interaction_quality, overall_score, recommendation, source, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.user_id,
                analysis.username,
                analysis.analyzed_at.isoformat(),
                analysis.engagement_rate,
                analysis.topic_alignment,
                analysis.interaction_quality,
                analysis.overall_score,
                analysis.recommendation,
                analysis.source.value,
                analysis.notes
            ))
            
            # Update target user status
            c.execute('''
                UPDATE target_users
                SET status = 'analyzed'
                WHERE user_id = ?
            ''', (analysis.user_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            return False
    
    def get_analysis_history(self, user_id: str) -> Optional[Dict]:
        """Get analysis history for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                SELECT * FROM analyzed_users
                WHERE user_id = ?
                ORDER BY last_analyzed DESC
                LIMIT 1
            ''', (user_id,))
            
            row = c.fetchone()
            if row:
                return {
                    'user_id': row[0],
                    'username': row[1],
                    'last_analyzed': row[2],
                    'engagement_rate': row[3],
                    'topic_alignment': row[4],
                    'interaction_quality': row[5],
                    'overall_score': row[6],
                    'recommendation': row[7],
                    'source': row[8],
                    'notes': row[9]
                }
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error getting analysis history: {str(e)}")
            return None 