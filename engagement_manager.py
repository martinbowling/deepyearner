"""
Sophisticated engagement manager that handles conversation heuristics.
Uses unified MemorySystem interface and configurable settings.
"""
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import yaml
import os
from dataclasses import dataclass
import json

from memory_system import MemorySystem, Memory
from twitter_utils import TwitterClient

logger = logging.getLogger(__name__)

@dataclass
class ConversationMetrics:
    """Metrics for a conversation"""
    start_time: datetime
    last_update: datetime
    reply_count: int
    participants: Set[str]
    sentiment_history: List[float]
    engagement_scores: List[float]
    topic_relevance: float
    user_values: Dict[str, float]
    momentum_score: float

class EngagementManager:
    """Manages conversation engagement using sophisticated heuristics"""
    
    def __init__(
        self,
        twitter_client: TwitterClient,
        memory_system: MemorySystem,
        config_path: str
    ):
        self.twitter = twitter_client
        self.memory = memory_system
        self.settings = self._load_settings(config_path)
        self.active_conversations: Dict[str, ConversationMetrics] = {}
        self.last_cleanup = datetime.now()
    
    def _load_settings(self, config_path: str) -> Dict:
        """Load engagement settings from YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading engagement settings: {str(e)}")
            return {}
    
    def should_engage(
        self,
        tweet: Dict,
        context: Dict
    ) -> tuple[bool, float, str]:
        """Determine if we should engage with a tweet"""
        try:
            # Calculate component scores
            user_score = self._calculate_user_value(tweet, context)
            content_score = self._calculate_content_quality(tweet, context)
            timeliness_score = self._calculate_timeliness(tweet)
            momentum_score = self._calculate_momentum(tweet, context)
            
            # Weight and combine scores
            weights = self.settings['scoring_weights']
            total_score = (
                user_score * weights['user_value'] +
                content_score * weights['content_quality'] +
                timeliness_score * weights['timeliness'] +
                momentum_score * weights['conversation_momentum']
            )
            
            # Check termination triggers
            if self._check_termination_triggers(tweet, context):
                return False, total_score, "Termination trigger activated"
            
            # Check continuation triggers
            force_continue = self._check_continuation_triggers(tweet, context)
            
            # Make decision
            should_engage = (
                total_score > 0.6 or
                force_continue or
                self._is_high_value_conversation(tweet, context)
            )
            
            reason = self._get_engagement_reason(
                should_engage,
                total_score,
                force_continue
            )
            
            return should_engage, total_score, reason
            
        except Exception as e:
            logger.error(f"Error evaluating engagement: {str(e)}")
            return False, 0.0, "Error in evaluation"
    
    def update_conversation_metrics(
        self,
        conversation_id: str,
        tweet: Dict,
        context: Dict
    ):
        """Update metrics for a conversation"""
        try:
            if conversation_id not in self.active_conversations:
                # Initialize new conversation metrics
                self.active_conversations[conversation_id] = ConversationMetrics(
                    start_time=datetime.now(),
                    last_update=datetime.now(),
                    reply_count=1,
                    participants={tweet['author_id']},
                    sentiment_history=[],
                    engagement_scores=[],
                    topic_relevance=self._calculate_topic_relevance(tweet, context),
                    user_values={},
                    momentum_score=0.5
                )
            
            metrics = self.active_conversations[conversation_id]
            
            # Update basic metrics
            metrics.last_update = datetime.now()
            metrics.reply_count += 1
            metrics.participants.add(tweet['author_id'])
            
            # Update sentiment history
            sentiment = context.get('sentiment', 0.0)
            metrics.sentiment_history.append(sentiment)
            
            # Calculate and update engagement score
            engagement_score = self._calculate_engagement_score(tweet, context)
            metrics.engagement_scores.append(engagement_score)
            
            # Update user values
            user_value = self._calculate_user_value(tweet, context)
            metrics.user_values[tweet['author_id']] = user_value
            
            # Update momentum
            metrics.momentum_score = self._calculate_momentum(tweet, context)
            
            # Store metrics in memory
            self.memory.add_memory(Memory(
                id=f"conversation_metrics_{conversation_id}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                type='conversation_metrics',
                content={
                    'metrics': {
                        'reply_count': metrics.reply_count,
                        'participant_count': len(metrics.participants),
                        'avg_sentiment': sum(metrics.sentiment_history) / len(metrics.sentiment_history),
                        'avg_engagement': sum(metrics.engagement_scores) / len(metrics.engagement_scores),
                        'momentum': metrics.momentum_score,
                        'topic_relevance': metrics.topic_relevance
                    },
                    'conversation_id': conversation_id
                },
                context=context
            ))
            
        except Exception as e:
            logger.error(f"Error updating conversation metrics: {str(e)}")
    
    def _calculate_user_value(self, tweet: Dict, context: Dict) -> float:
        """Calculate user value score"""
        try:
            base_score = 0.5
            thresholds = self.settings['user_value']
            multipliers = thresholds['high_value_multipliers']
            
            # Check follower count
            if tweet.get('author_followers_count', 0) >= thresholds['min_follower_count']:
                base_score += 0.2
            
            # Check engagement rate
            if tweet.get('author_engagement_rate', 0) >= thresholds['min_engagement_rate']:
                base_score += 0.2
            
            # Apply multipliers
            if tweet.get('author_verified', False):
                base_score *= multipliers['verified']
            
            if tweet.get('author_is_mutual', False):
                base_score *= multipliers['mutual_follow']
            
            # Check previous interactions
            user_history = context.get('user_history', [])
            if user_history:
                positive_interactions = sum(
                    1 for h in user_history
                    if h.get('sentiment', 0) > 0.5
                )
                if positive_interactions > 2:
                    base_score *= multipliers['previous_positive_interactions']
            
            # Check expertise
            if self._is_expert_in_field(tweet['author_id'], context):
                base_score *= multipliers['expert_in_field']
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating user value: {str(e)}")
            return 0.5
    
    def _calculate_content_quality(self, tweet: Dict, context: Dict) -> float:
        """Calculate content quality score"""
        try:
            thresholds = self.settings['quality_thresholds']
            
            # Check sentiment
            sentiment = context.get('sentiment', 0.0)
            if sentiment < thresholds['min_sentiment']:
                return 0.0
            
            # Check relevance
            relevance = self._calculate_topic_relevance(tweet, context)
            if relevance < thresholds['min_relevance']:
                return 0.0
            
            # Check coherence
            coherence = context.get('coherence', 0.0)
            if coherence < thresholds['min_coherence']:
                return 0.0
            
            # Check toxicity
            toxicity = context.get('toxicity', 0.0)
            if toxicity > thresholds['toxic_threshold']:
                return 0.0
            
            # Calculate final score
            return min(
                1.0,
                (
                    (sentiment + 1) * 0.3 +  # Convert -1,1 to 0,1
                    relevance * 0.4 +
                    coherence * 0.3
                )
            )
            
        except Exception as e:
            logger.error(f"Error calculating content quality: {str(e)}")
            return 0.5
    
    def _calculate_timeliness(self, tweet: Dict) -> float:
        """Calculate timeliness score"""
        try:
            current_hour = datetime.now().hour
            peak_hours = self.settings['time_limits']['peak_hours']
            
            # Base score for peak hours
            if peak_hours[0] <= current_hour <= peak_hours[1]:
                base_score = 0.8
            else:
                base_score = 0.5
            
            # Check tweet age
            tweet_age = datetime.now() - datetime.fromisoformat(tweet['created_at'])
            max_age = timedelta(hours=self.settings['time_limits']['max_conversation_age_hours'])
            
            if tweet_age > max_age:
                return 0.0
            
            # Decay score based on age
            age_factor = 1.0 - (tweet_age.total_seconds() / max_age.total_seconds())
            
            return base_score * age_factor
            
        except Exception as e:
            logger.error(f"Error calculating timeliness: {str(e)}")
            return 0.5
    
    def _calculate_momentum(self, tweet: Dict, context: Dict) -> float:
        """Calculate conversation momentum score"""
        try:
            weights = self.settings['momentum_factors']
            
            # Calculate reply speed
            reply_speed = self._calculate_reply_speed(tweet, context)
            
            # Calculate engagement growth
            engagement_growth = self._calculate_engagement_growth(tweet, context)
            
            # Calculate topic relevance trend
            topic_trend = self._calculate_topic_relevance(tweet, context)
            
            return min(
                1.0,
                reply_speed * weights['reply_speed'] +
                engagement_growth * weights['engagement_growth'] +
                topic_trend * weights['topic_relevance']
            )
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return 0.5
    
    def _check_termination_triggers(self, tweet: Dict, context: Dict) -> bool:
        """Check if any termination triggers are active"""
        try:
            triggers = self.settings['terminate_if']
            
            # Check toxic content
            if (
                triggers['toxic_content_detected'] and
                context.get('toxicity', 0.0) > self.settings['quality_thresholds']['toxic_threshold']
            ):
                return True
            
            # Check spam patterns
            if (
                triggers['spam_patterns_detected'] and
                self._detect_spam_patterns(tweet, context)
            ):
                return True
            
            # Check engagement decline
            if (
                triggers['engagement_declining'] and
                self._is_engagement_declining(tweet, context)
            ):
                return True
            
            # Check topic drift
            if (
                triggers['off_topic_drift'] and
                self._calculate_topic_relevance(tweet, context) < 0.3
            ):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking termination triggers: {str(e)}")
            return True
    
    def _check_continuation_triggers(self, tweet: Dict, context: Dict) -> bool:
        """Check if any continuation triggers are active"""
        try:
            triggers = self.settings['continue_if']
            
            # Check high-value user engagement
            if (
                triggers['high_value_user_engaged'] and
                self._calculate_user_value(tweet, context) > 0.8
            ):
                return True
            
            # Check growing engagement
            if (
                triggers['growing_engagement'] and
                self._is_engagement_growing(tweet, context)
            ):
                return True
            
            # Check trending topic
            if (
                triggers['trending_topic'] and
                self._is_topic_trending(tweet, context)
            ):
                return True
            
            # Check sentiment trend
            if (
                triggers['positive_sentiment_trend'] and
                self._is_sentiment_improving(tweet, context)
            ):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking continuation triggers: {str(e)}")
            return False
    
    def _is_high_value_conversation(self, tweet: Dict, context: Dict) -> bool:
        """Check if this is a high-value conversation"""
        try:
            conversation_id = tweet.get('conversation_id')
            if not conversation_id:
                return False
            
            metrics = self.active_conversations.get(conversation_id)
            if not metrics:
                return False
            
            # Check participant value
            high_value_participants = sum(
                1 for value in metrics.user_values.values()
                if value > 0.8
            )
            
            # Check engagement trend
            engagement_growing = (
                len(metrics.engagement_scores) >= 3 and
                metrics.engagement_scores[-1] > metrics.engagement_scores[-3]
            )
            
            # Check sentiment trend
            sentiment_positive = (
                len(metrics.sentiment_history) >= 3 and
                sum(metrics.sentiment_history[-3:]) / 3 > 0.5
            )
            
            return (
                high_value_participants >= 2 or
                (engagement_growing and sentiment_positive)
            )
            
        except Exception as e:
            logger.error(f"Error checking conversation value: {str(e)}")
            return False
    
    def _get_engagement_reason(
        self,
        should_engage: bool,
        score: float,
        force_continue: bool
    ) -> str:
        """Get human-readable reason for engagement decision"""
        if not should_engage:
            if score < 0.6:
                return "Engagement score too low"
            return "Termination trigger activated"
        
        if force_continue:
            return "Continuation trigger activated"
        
        if score > 0.8:
            return "High-value engagement opportunity"
        
        return "Standard engagement criteria met"
    
    def cleanup_conversations(self):
        """Remove old conversations and update metrics"""
        try:
            now = datetime.now()
            if now - self.last_cleanup < timedelta(minutes=30):
                return
            
            to_remove = []
            for conv_id, metrics in self.active_conversations.items():
                # Remove old conversations
                age = now - metrics.start_time
                if age > timedelta(hours=self.settings['time_limits']['max_conversation_age_hours']):
                    to_remove.append(conv_id)
                    continue
                
                # Remove inactive conversations
                inactive_time = now - metrics.last_update
                if inactive_time > timedelta(hours=2):
                    to_remove.append(conv_id)
                    continue
            
            # Remove conversations
            for conv_id in to_remove:
                metrics = self.active_conversations[conv_id]
                
                # Store final metrics in memory
                self.memory.add_memory(Memory(
                    id=f"conversation_end_{conv_id}_{now.timestamp()}",
                    timestamp=now,
                    type='conversation_end',
                    content={
                        'duration': (now - metrics.start_time).total_seconds(),
                        'final_metrics': {
                            'reply_count': metrics.reply_count,
                            'participant_count': len(metrics.participants),
                            'avg_sentiment': sum(metrics.sentiment_history) / len(metrics.sentiment_history),
                            'avg_engagement': sum(metrics.engagement_scores) / len(metrics.engagement_scores),
                            'final_momentum': metrics.momentum_score
                        }
                    },
                    context={'conversation_id': conv_id}
                ))
                
                del self.active_conversations[conv_id]
            
            self.last_cleanup = now
            
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {str(e)}")
    
    def _is_expert_in_field(self, user_id: str, context: Dict) -> bool:
        """Determine if user is an expert in relevant fields"""
        try:
            # Get user's topic expertise from memory
            expertise_memories = self.memory.get_memories_by_type(
                'user_expertise',
                user_id=user_id
            )
            
            if not expertise_memories:
                return False
            
            # Get conversation topics
            conversation_topics = set(context.get('topics', []))
            
            # Check topic overlap
            for memory in expertise_memories:
                expert_topics = set(memory.content.get('topics', []))
                if expert_topics & conversation_topics:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking user expertise: {str(e)}")
            return False
