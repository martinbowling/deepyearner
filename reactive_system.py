"""
Reactive system that handles mentions, replies, and other interactions.
Uses unified TwitterClient and MemorySystem interfaces.
"""
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import json

from twitter_utils import TwitterClient
from memory_system import MemorySystem, Memory
from prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class ConversationChain:
    """Tracks an ongoing conversation"""
    def __init__(self, initial_tweet: Dict):
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.tweets = [initial_tweet]
        self.participants = {initial_tweet['author_id']}
        self.our_last_reply = None
        self.topics = set()
        self.context = {}
        
    def add_tweet(self, tweet: Dict):
        """Add a tweet to the conversation"""
        self.tweets.append(tweet)
        self.participants.add(tweet['author_id'])
        self.last_update = datetime.now()

class ReactiveSystem:
    """Handles reactive engagement with mentions, replies, and conversations"""
    
    def __init__(
        self,
        twitter_client: TwitterClient,
        memory_system: MemorySystem,
        prompt_manager: PromptManager
    ):
        self.twitter = twitter_client
        self.memory = memory_system
        self.prompt_manager = prompt_manager
        self.active_conversations: Dict[str, ConversationChain] = {}
        self.last_mention_id = None
        self.last_check = datetime.now()
        
    async def check_mentions(self):
        """Check for new mentions and handle them"""
        try:
            # Get mentions since last check
            mentions = self.twitter.get_user_mentions(
                self.twitter.user_id,
                since_id=self.last_mention_id
            )
            
            if not mentions:
                return
            
            # Update last mention ID
            self.last_mention_id = max(
                mention['id'] for mention in mentions
            )
            
            # Handle each mention
            for mention in mentions:
                await self.handle_mention(mention)
                
        except Exception as e:
            logger.error(f"Error checking mentions: {str(e)}")
    
    async def handle_mention(self, mention: Dict):
        """Handle a new mention or reply"""
        try:
            # Get conversation context
            context = await self.get_conversation_context(mention)
            
            # Find or create conversation
            conversation = self.find_or_create_conversation(mention, context)
            
            # Evaluate engagement
            engagement = await self.evaluate_engagement(mention, context)
            
            if engagement['should_engage']:
                # Generate response
                response = await self.generate_response(mention, context)
                
                if response:
                    # Take action based on engagement type
                    action_taken = False
                    
                    if engagement['engagement_type'] == 'reply':
                        tweet_id = self.twitter.create_tweet(
                            text=response['text'],
                            reply_to_id=mention['id']
                        )
                        if tweet_id:
                            action_taken = True
                            conversation.our_last_reply = tweet_id
                            conversation.add_tweet({
                                'id': tweet_id,
                                'text': response['text'],
                                'author_id': self.twitter.user_id
                            })
                            
                    elif engagement['engagement_type'] == 'quote':
                        tweet_id = self.twitter.create_tweet(
                            text=response['text'],
                            quote_tweet_id=mention['id']
                        )
                        if tweet_id:
                            action_taken = True
                            conversation.our_last_reply = tweet_id
                            
                    elif engagement['engagement_type'] == 'retweet':
                        if self.twitter.retweet(mention['id']):
                            action_taken = True
                    
                    # Store interaction in memory if action was taken
                    if action_taken:
                        self.memory.add_memory(Memory(
                            id=f"interaction_{datetime.now().timestamp()}",
                            timestamp=datetime.now(),
                            type='interaction',
                            content={
                                'mention': mention,
                                'response': response,
                                'engagement_type': engagement['engagement_type']
                            },
                            context=context,
                            participants=conversation.participants
                        ))
                        
        except Exception as e:
            logger.error(f"Error handling mention {mention['id']}: {str(e)}")
    
    async def get_conversation_context(self, tweet: Dict) -> Dict:
        """Get context for a conversation"""
        try:
            context = {
                'conversation': [],
                'user_history': [],
                'topics': set(),
                'sentiment': 0.0
            }
            
            # Get conversation history
            if 'conversation_id' in tweet:
                conversation_memories = self.memory.get_conversation_context(
                    tweet['conversation_id']
                )
                context['conversation'] = [
                    m.content for m in conversation_memories
                ]
            
            # Get user interaction history
            user_memories = self.memory.get_user_interaction_history(
                tweet['author_id']
            )
            context['user_history'] = [
                m.content for m in user_memories
            ]
            
            # Extract topics from memory
            for memory in conversation_memories + user_memories:
                if 'topics' in memory.content:
                    context['topics'].update(memory.content['topics'])
            
            # Get personality state
            context['personality_state'] = self.memory.get_personality_state()
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return {}
    
    def find_or_create_conversation(
        self,
        tweet: Dict,
        context: Dict
    ) -> ConversationChain:
        """Find existing conversation or create new one"""
        conversation_id = tweet.get('conversation_id', tweet['id'])
        
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = ConversationChain(tweet)
            
            # Add topics from context
            if 'topics' in context:
                self.active_conversations[conversation_id].topics.update(
                    context['topics']
                )
        
        return self.active_conversations[conversation_id]
    
    async def evaluate_engagement(self, tweet: Dict, context: Dict) -> Dict:
        """Evaluate whether and how to engage with a tweet"""
        try:
            # Get evaluation prompt
            prompt = self.prompt_manager.get_engagement_prompt(tweet, context)
            
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Make decision based on prompt and state
            decision = {
                'should_engage': personality_state['energy'] > 0.3,
                'engagement_type': 'reply',  # Default to reply
                'confidence': 0.7
            }
            
            # Adjust based on tweet content and context
            if 'quote' in tweet.get('text', '').lower():
                decision['engagement_type'] = 'quote'
            elif personality_state['energy'] < 0.5:
                decision['engagement_type'] = 'retweet'
            
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating engagement: {str(e)}")
            return {'should_engage': False}
    
    async def generate_response(self, tweet: Dict, context: Dict) -> Optional[Dict]:
        """Generate a response to a tweet"""
        try:
            # Get response prompt
            prompt = self.prompt_manager.get_response_prompt(tweet, context)
            
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Generate response based on prompt and state
            # This would typically use an LLM
            response = {
                'text': f"Thanks for your tweet! {tweet['text'][:50]}...",
                'topics': context.get('topics', set()),
                'sentiment': 0.7
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return None
    
    def cleanup_conversations(self):
        """Remove old conversations"""
        try:
            now = datetime.now()
            to_remove = []
            
            for conv_id, conv in self.active_conversations.items():
                # Remove conversations inactive for 6 hours
                if now - conv.last_update > timedelta(hours=6):
                    to_remove.append(conv_id)
            
            for conv_id in to_remove:
                del self.active_conversations[conv_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {str(e)}")
