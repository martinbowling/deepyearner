"""
Continuous bot runner that generates and posts content over time.
Includes monitoring and rate limiting.
"""
import asyncio
import logging
from datetime import datetime, timedelta
import random
from typing import Dict, List
import os
import sqlite3
from dotenv import load_dotenv

from content_generator import ContentGenerator
from twitter_utils import TwitterClient
from memory_system import MemorySystem
from prompt_manager import PromptManager
import anthropic

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BotRunner:
    def __init__(
        self,
        content_generator: ContentGenerator,
        min_interval_minutes: int = 30,
        max_interval_minutes: int = 120
    ):
        self.generator = content_generator
        self.min_interval = min_interval_minutes
        self.max_interval = max_interval_minutes
        self.last_post_time = None
        self.running = False

    async def start(self):
        """Start the bot running continuously"""
        self.running = True
        logger.info("Starting continuous bot run...")
        
        while self.running:
            try:
                # Get current context
                current_context = await self._get_current_context()
                
                # Generate content suggestions
                suggestions = await self.generator.generate_content_suggestions(
                    current_context,
                    max_suggestions=3
                )
                
                if suggestions:
                    # Pick best suggestion
                    best_suggestion = max(
                        suggestions,
                        key=lambda x: (
                            x.confidence * 0.3 +
                            x.personality_alignment * 0.3 +
                            x.expected_engagement * 0.4
                        )
                    )
                    
                    if best_suggestion.confidence > 0.7:
                        # Post as thread if complex enough
                        if len(best_suggestion.text.split()) > 20:
                            await self.generator.post_thread(
                                best_suggestion.topics.pop(),
                                [{"text": best_suggestion.text}],
                                max_tweets=3
                            )
                        else:
                            # Post as single tweet
                            await self.generator.twitter.post_tweet(best_suggestion.text)
                        
                        self.last_post_time = datetime.now()
                        logger.info(f"Posted content at {self.last_post_time}")
                
                # Random sleep interval
                sleep_minutes = random.randint(self.min_interval, self.max_interval)
                logger.info(f"Sleeping for {sleep_minutes} minutes...")
                await asyncio.sleep(sleep_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in bot run loop: {str(e)}")
                # Sleep for a bit before retrying
                await asyncio.sleep(300)

    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("Stopping bot...")

    async def _get_current_context(self) -> Dict:
        """Get current context for content generation"""
        try:
            # Get timeline
            timeline = await self.generator.twitter.get_home_timeline()
            
            # Get time of day context
            hour = datetime.now().hour
            time_context = {
                'early_morning': 5 <= hour <= 8,
                'business_hours': 9 <= hour <= 17,
                'evening': 18 <= hour <= 23,
                'late_night': hour <= 4
            }
            
            return {
                'timeline': timeline,
                'time_context': time_context,
                'last_post_time': self.last_post_time
            }
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return {}

async def main():
    # Create directories if they don't exist
    os.makedirs("data/memory", exist_ok=True)
    os.makedirs("data/vectors", exist_ok=True)
    
    # Initialize SQLite database
    db = sqlite3.connect("data/memory/bot.db")
    
    # Initialize components in correct order
    twitter = TwitterClient()
    memory = MemorySystem(
        db=db,
        persist_directory="data/vectors"
    )
    # Initialize PromptManager with memory_system
    prompts = PromptManager(memory_system=memory)
    
    # Initialize Anthropic client
    client = anthropic.Client(
        api_key=os.getenv("ANTHROPIC_API_KEY")  # Make sure this env var is set
    )
    
    generator = ContentGenerator(
        twitter_client=twitter,
        memory_system=memory,
        prompt_manager=prompts,
        anthropic_client=client
    )
    
    # Create and start bot runner
    runner = BotRunner(generator)
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        runner.stop()
        logger.info("Bot stopped by user")
        db.close()
    except Exception as e:
        logger.error(f"Bot stopped due to error: {str(e)}")
        runner.stop()
        db.close()

if __name__ == "__main__":
    asyncio.run(main()) 