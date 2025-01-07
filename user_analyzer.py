"""
Analyzer for evaluating Twitter users as potential follows.
"""
from typing import Dict, List, Optional
from datetime import datetime
import logging
from user_analysis import UserAnalysis, AnalysisSource
from twitter_utils import TwitterClient
import time

logger = logging.getLogger(__name__)

class UserAnalyzer:
    """Analyzes Twitter users for follow potential"""
    
    def __init__(self, twitter_client: TwitterClient):
        """Initialize with Twitter client"""
        self.twitter = twitter_client
    
    def analyze_user(self, user_id: str, username: str, source: AnalysisSource) -> Optional[UserAnalysis]:
        """Analyze a user and return results"""
        try:
            # Get user profile
            user = self.twitter.get_user_by_id(user_id)
            if not user:
                return None
                
            # Get recent tweets (up to 100, the maximum per request)
            tweets = self.twitter.get_users_tweets(user_id, max_results=100)
            if not tweets or 'data' not in tweets:
                return None
            
            # Calculate engagement metrics
            engagement_rate = self._calculate_engagement(tweets['data'])
            
            # Analyze tweet content and topics
            topic_alignment = self._analyze_topics(tweets['data'])
            
            # Analyze interaction quality
            interaction_quality = self._analyze_interactions(tweets['data'])
            
            # Calculate overall score
            overall_score = (
                engagement_rate * 0.4 +
                topic_alignment * 0.4 +
                interaction_quality * 0.2
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(overall_score)
            
            # Generate analysis notes
            notes = self._generate_notes(
                engagement_rate,
                topic_alignment,
                interaction_quality,
                tweets['data']
            )
            
            # Automatically follow if highly recommended
            if overall_score >= 0.8:
                logger.info(f"Auto-following highly recommended user {username}")
                if self.twitter.follow_user(user_id):
                    notes = f"Auto-followed user. {notes}"
            # Consider following if recommended with good engagement
            elif overall_score >= 0.6 and engagement_rate >= 0.7:
                logger.info(f"Auto-following recommended user {username} with high engagement")
                if self.twitter.follow_user(user_id):
                    notes = f"Auto-followed user based on high engagement. {notes}"
            
            return UserAnalysis(
                user_id=user_id,
                username=username,
                analyzed_at=datetime.now(),
                engagement_rate=engagement_rate,
                topic_alignment=topic_alignment,
                interaction_quality=interaction_quality,
                overall_score=overall_score,
                recommendation=recommendation,
                source=source,
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"Error analyzing user {username}: {str(e)}")
            return None
    
    def _calculate_engagement(self, tweets: List[Dict]) -> float:
        """Calculate engagement rate from tweets"""
        if not tweets:
            return 0.0
            
        total_engagement = 0
        for tweet in tweets:
            metrics = tweet.get('public_metrics', {})
            likes = metrics.get('like_count', 0)
            retweets = metrics.get('retweet_count', 0)
            replies = metrics.get('reply_count', 0)
            quotes = metrics.get('quote_count', 0)
            
            # Weight different types of engagement
            weighted_engagement = (
                likes * 1.0 +
                retweets * 1.5 +
                replies * 2.0 +
                quotes * 1.5
            )
            
            total_engagement += weighted_engagement
        
        # Calculate average engagement per tweet
        avg_engagement = total_engagement / len(tweets)
        
        # Normalize to 0-1 scale (assuming 50+ is very high engagement)
        return min(1.0, avg_engagement / 50)
    
    def _analyze_topics(self, tweets: List[Dict]) -> float:
        """Analyze tweet topics and calculate alignment score"""
        # TODO: Implement more sophisticated topic analysis
        # For now, return a placeholder score
        return 0.7
    
    def _analyze_interactions(self, tweets: List[Dict]) -> float:
        """Analyze quality of user interactions"""
        # TODO: Implement interaction quality analysis
        # For now, return a placeholder score
        return 0.6
    
    def _generate_recommendation(self, overall_score: float) -> str:
        """Generate follow recommendation based on score"""
        if overall_score >= 0.8:
            return "Highly Recommended"
        elif overall_score >= 0.6:
            return "Recommended"
        elif overall_score >= 0.4:
            return "Consider"
        else:
            return "Not Recommended"
    
    def _generate_notes(self, engagement: float, topics: float, 
                       interactions: float, tweets: List[Dict]) -> str:
        """Generate analysis notes"""
        notes = []
        
        # Engagement notes
        if engagement >= 0.8:
            notes.append("Very high engagement rate")
        elif engagement >= 0.5:
            notes.append("Good engagement rate")
        else:
            notes.append("Low engagement rate")
            
        # Topic notes
        if topics >= 0.8:
            notes.append("Strong topic alignment")
        elif topics >= 0.5:
            notes.append("Moderate topic alignment")
        else:
            notes.append("Low topic alignment")
            
        # Interaction notes
        if interactions >= 0.8:
            notes.append("High quality interactions")
        elif interactions >= 0.5:
            notes.append("Decent interaction quality")
        else:
            notes.append("Poor interaction quality")
        
        return " | ".join(notes) 