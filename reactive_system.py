from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from datetime import datetime, timedelta
import json

from twitter_utils import TwitterAPI
from prompt_manager import PromptManager
from personality_state import PersonalityState
from twitter_actions import TwitterActions
from memory_system import MemorySystem

logger = logging.getLogger(__name__)

class ConversationChain:
    """Tracks an ongoing conversation thread"""
    def __init__(self, root_tweet_id: str, our_tweet_id: Optional[str] = None):
        self.root_tweet_id = root_tweet_id
        self.our_tweet_id = our_tweet_id
        self.replies: List[Dict] = []
        self.participants: Dict[str, int] = {}
        self.last_activity = datetime.now()
        self.our_last_reply = None
        self.engagement_metrics = {
            'replies': 0,
            'likes': 0,
            'retweets': 0,
            'quality_score': 0.0
        }
        
    def add_reply(self, reply: Dict):
        """Add a reply to the conversation chain"""
        self.replies.append(reply)
        author_id = reply['author_id']
        self.participants[author_id] = self.participants.get(author_id, 0) + 1
        self.last_activity = datetime.now()
        
    def should_continue(self) -> bool:
        """Determine if we should continue engaging in this conversation"""
        # Check if conversation is stale
        if datetime.now() - self.last_activity > timedelta(hours=24):
            return False
            
        # Check if we're too deep in the thread
        if len(self.replies) > 10:
            return False
            
        # Check if engagement is good
        if self.engagement_metrics['quality_score'] < 0.5:
            return False
            
        return True
        
    def update_metrics(self, metrics: Dict):
        """Update engagement metrics for the conversation"""
        self.engagement_metrics.update(metrics)
        # Calculate quality score based on engagement
        total_engagement = (
            metrics.get('replies', 0) * 2 +
            metrics.get('likes', 0) +
            metrics.get('retweets', 0) * 1.5
        )
        self.engagement_metrics['quality_score'] = min(total_engagement / 10.0, 1.0)

class ReactiveSystem:
    """Handles reactive engagement with mentions, replies, and conversations"""
    
    def __init__(
        self,
        twitter_api: TwitterAPI,
        twitter_actions: TwitterActions,
        personality: PersonalityState,
        prompt_manager: PromptManager,
        db: Any,
        vector_store: Any,
        memory_system: MemorySystem
    ):
        self.twitter_api = twitter_api
        self.twitter_actions = twitter_actions
        self.personality = personality
        self.prompt_manager = prompt_manager
        self.db = db
        self.vector_store = vector_store
        self.memory_system = memory_system
        self.active_conversations: Dict[str, ConversationChain] = {}
        self.last_mention_id = None
        self.last_check = datetime.now()
        
    async def process_new_interactions(self):
        """Process new mentions and replies"""
        try:
            # Get new mentions since last check
            mentions = self.twitter_api.get_mentions_timeline(
                since_id=self.last_mention_id
            )
            
            if mentions:
                self.last_mention_id = mentions[0]['id']
                
            # Process each mention
            for mention in mentions:
                await self.handle_mention(mention)
                
            # Update active conversations
            await self.update_conversations()
            
            # Clean up stale conversations
            self.cleanup_conversations()
            
            self.last_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing interactions: {str(e)}")
            
    async def handle_mention(self, mention: Dict):
        """Handle a new mention or reply"""
        try:
            # Get conversation context
            context = await self.get_conversation_context(mention)
            
            # Check if part of existing conversation
            conversation = self.find_or_create_conversation(mention, context)
            
            # Evaluate if and how to engage
            engagement = await self.evaluate_engagement(mention, context)
            
            if engagement['should_engage']:
                # Generate response
                response = await self.generate_response(mention, context)
                
                if response:
                    # Determine engagement type
                    if engagement['engagement_type'] == 'reply':
                        # Post reply
                        reply_id = self.twitter_actions.reply_to_tweet(
                            mention['id'],
                            response['response_text']
                        )
                        
                        if reply_id:
                            # Update conversation
                            conversation.our_last_reply = reply_id
                            conversation.add_reply({
                                'id': reply_id,
                                'text': response['response_text'],
                                'author_id': self.twitter_api.user_id
                            })
                            
                    elif engagement['engagement_type'] == 'quote':
                        # Post quote tweet
                        quote_id = self.twitter_actions.quote_tweet(
                            mention['id'],
                            response['response_text']
                        )
                        
                        if quote_id:
                            conversation.our_last_reply = quote_id
                            
                    elif engagement['engagement_type'] == 'retweet':
                        # Retweet
                        self.twitter_actions.retweet(mention['id'])
                        
                    # Store interaction
                    self.store_interaction(
                        mention,
                        response,
                        engagement['engagement_type']
                    )
                    
        except Exception as e:
            logger.error(f"Error handling mention {mention['id']}: {str(e)}")
            
    async def get_conversation_context(self, tweet: Dict) -> Dict:
        """Get full context for a conversation"""
        try:
            # Get conversation thread
            conversation = self.twitter_api.get_conversation_thread(tweet['id'])
            
            # Get author info
            author = self.twitter_api.get_user_by_id(tweet['author_id'])
            
            # Get interaction history
            history = self.get_interaction_history(tweet['author_id'])
            
            # Get relevant memory
            memory = await self.get_relevant_memory(tweet, conversation)
            
            return {
                'conversation': conversation,
                'author': author,
                'history': history,
                'memory': memory,
                'tweet': tweet
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return {'tweet': tweet}
            
    def find_or_create_conversation(
        self,
        tweet: Dict,
        context: Dict
    ) -> ConversationChain:
        """Find existing conversation or create new one"""
        # Check if reply to our tweet
        if tweet.get('in_reply_to_user_id') == self.twitter_api.user_id:
            root_id = tweet.get('in_reply_to_status_id')
            if root_id in self.active_conversations:
                conversation = self.active_conversations[root_id]
                conversation.add_reply(tweet)
                return conversation
                
        # Check if part of existing conversation
        for conv in self.active_conversations.values():
            if tweet['id'] in [r['id'] for r in conv.replies]:
                return conv
                
        # Create new conversation
        conversation = ConversationChain(tweet['id'])
        self.active_conversations[tweet['id']] = conversation
        return conversation
        
    async def evaluate_engagement(self, tweet: Dict, context: Dict) -> Dict:
        """Evaluate whether and how to engage with a tweet"""
        try:
            # Get evaluation prompt
            prompt = self.prompt_manager.get_contextualized_prompt(
                "should_I_reply",
                self.personality,
                context
            )
            
            # Get decision from Claude
            response = await self.personality.get_response(prompt)
            decision = json.loads(response)
            
            # Store decision for learning
            self.store_engagement_decision(tweet['id'], decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating engagement: {str(e)}")
            return {'should_engage': False}
            
    async def generate_response(
        self,
        tweet: Dict,
        context: Dict
    ) -> Optional[Dict]:
        """Generate a response to a tweet"""
        try:
            # Get response prompt
            prompt = self.prompt_manager.get_contextualized_prompt(
                "engagement_response",
                self.personality,
                context
            )
            
            # Get response from Claude
            response = await self.personality.get_response(prompt)
            
            # Extract response data
            response_data = json.loads(response)
            
            # Validate response
            if self.validate_response(response_data):
                return response_data
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return None
            
    async def update_conversations(self):
        """Update active conversations with new activity"""
        for conv_id, conversation in self.active_conversations.items():
            try:
                # Get new replies
                new_replies = self.twitter_api.get_tweet_replies(conv_id)
                
                # Update conversation
                for reply in new_replies:
                    if reply['id'] not in [r['id'] for r in conversation.replies]:
                        conversation.add_reply(reply)
                        
                # Update metrics
                metrics = self.twitter_api.get_tweet_metrics(conv_id)
                conversation.update_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error updating conversation {conv_id}: {str(e)}")
                
    def cleanup_conversations(self):
        """Remove stale conversations"""
        now = datetime.now()
        to_remove = []
        
        for conv_id, conversation in self.active_conversations.items():
            if now - conversation.last_activity > timedelta(days=2):
                to_remove.append(conv_id)
                
        for conv_id in to_remove:
            del self.active_conversations[conv_id]
            
    def get_interaction_history(self, user_id: str) -> List[Dict]:
        """Get history of interactions with a user"""
        return list(self.db["interactions"].rows_where(
            "author_id = ? AND timestamp > datetime('now', '-30 days')",
            [user_id],
            order_by="timestamp desc",
            limit=10
        ))
        
    async def get_relevant_memory(
        self,
        tweet: Dict,
        conversation: List[Dict]
    ) -> Dict:
        """Get relevant memory for the conversation"""
        # Combine tweet text with conversation
        text = tweet['text'] + "\n" + "\n".join(
            t['text'] for t in conversation
        )
        
        # Get similar content from vector store
        similar = self.vector_store.search(text, k=5)
        
        # Get active bits
        bits = self.get_active_bits()
        
        return {
            'similar_content': similar,
            'active_bits': bits
        }
        
    def get_active_bits(self) -> List[Dict]:
        """Get currently active bits and memes"""
        return list(self.db["meme_tracker"].rows_where(
            "success_rate > ? AND last_used > datetime('now', '-7 days')",
            [0.6],
            order_by="last_used desc",
            limit=5
        ))
        
    def store_interaction(
        self,
        mention: Dict,
        response_data: Dict,
        engagement_type: str
    ):
        """Store interaction in database"""
        self.db["interactions"].insert({
            "tweet_id": mention['id'],
            "author_id": mention['author_id'],
            "our_reply_id": response_data.get('id'),
            "response_data": json.dumps(response_data),
            "timestamp": datetime.now().isoformat(),
            "engagement_type": engagement_type
        })
        
    def store_engagement_decision(self, tweet_id: str, decision: Dict):
        """Store engagement decision for learning"""
        self.db["engagement_decisions"].insert({
            "tweet_id": tweet_id,
            "decision": json.dumps(decision),
            "timestamp": datetime.now().isoformat()
        })
        
    def validate_response(self, response_data: Dict) -> bool:
        """Validate generated response"""
        try:
            # Check confidence threshold
            if response_data['confidence_score'] < 0.7:
                return False
                
            # Check personality alignment
            personality_scores = response_data['personality_alignment']
            if any(score < 0.6 for score in personality_scores.values()):
                return False
                
            # Check risk assessment
            risk = response_data['risk_assessment']
            if risk['controversy_level'] > 0.7 or risk['potential_misinterpretation'] > 0.6:
                return False
                
            # Validate response length
            if len(response_data['response_text']) > 280:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            return False
