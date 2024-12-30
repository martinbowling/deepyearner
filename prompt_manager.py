"""
Prompt manager that handles prompt generation and templating.
Uses unified MemorySystem interface.
"""
import logging
from typing import Dict, Optional
from datetime import datetime
import json
import os

from memory_system import MemorySystem

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompts and templates for the bot"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from files"""
        templates = {}
        try:
            for filename in os.listdir(self.prompts_dir):
                if filename.endswith('.txt'):
                    template_name = filename[:-4]
                    with open(os.path.join(self.prompts_dir, filename)) as f:
                        templates[template_name] = f.read()
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
        return templates
    
    def get_engagement_prompt(
        self,
        tweet: Dict,
        context: Dict
    ) -> str:
        """Get prompt for evaluating engagement"""
        try:
            template = self.templates.get('should_I_reply', '')
            if not template:
                return ''
            
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Format template with context
            return template.format(
                tweet_text=tweet['text'],
                author_history=json.dumps(context.get('user_history', [])),
                conversation_history=json.dumps(context.get('conversation', [])),
                personality_state=json.dumps(personality_state),
                current_time=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error getting engagement prompt: {str(e)}")
            return ''
    
    def get_response_prompt(
        self,
        tweet: Dict,
        context: Dict
    ) -> str:
        """Get prompt for generating a response"""
        try:
            template = self.templates.get('engagement_response', '')
            if not template:
                return ''
            
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Format template with context
            return template.format(
                tweet_text=tweet['text'],
                author_history=json.dumps(context.get('user_history', [])),
                conversation_history=json.dumps(context.get('conversation', [])),
                personality_state=json.dumps(personality_state),
                topics=json.dumps(list(context.get('topics', set()))),
                current_time=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error getting response prompt: {str(e)}")
            return ''
    
    def get_content_generation_prompt(
        self,
        context: Dict,
        content_type: str
    ) -> str:
        """Get prompt for generating content"""
        try:
            template = self.templates.get(f'generate_{content_type}', '')
            if not template:
                return ''
            
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Format template with context
            return template.format(
                personality_state=json.dumps(personality_state),
                recent_memories=json.dumps(context.get('recent_memories', [])),
                topics=json.dumps(list(context.get('topics', set()))),
                current_time=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error getting content generation prompt: {str(e)}")
            return ''
    
    def get_thread_prompt(
        self,
        topic: str,
        content: Dict,
        context: Dict
    ) -> str:
        """Get prompt for generating a thread"""
        try:
            template = self.templates.get('generate_thread', '')
            if not template:
                return ''
            
            # Get personality state
            personality_state = self.memory.get_personality_state()
            
            # Format template with context
            return template.format(
                topic=topic,
                relevant_content=json.dumps(content),
                personality_state=json.dumps(personality_state),
                current_time=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error getting thread prompt: {str(e)}")
            return ''
