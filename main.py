#!/usr/bin/env python3

import os
import sys
import json
import click
import asyncio
import logging
import traceback
from datetime import datetime
import sqlite_utils
from typing import Dict, List, Optional
import config
from twitter_utils import get_twitter_client
from memory_system import MemorySystem
from personality_system import PersonalitySystem
from vector_store import VectorStore
from pathlib import Path
import time
from anthropic import Anthropic

# Import configuration
from config import VECTOR_DB_PATH, ANTHROPIC_API_KEY

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_timeline(twitter_api):
    """Analyze the current timeline"""
    try:
        # Get recent tweets from our timeline
        tweets = twitter_api.get_users_tweets(max_results=20)
        if not tweets or 'data' not in tweets:
            return {
                "current_vibe": {"mood": "neutral", "energy_level": 0.5, "chaos_factor": 0.0},
                "trend_direction": {"trend": "stable"},
                "poast_opportunity": 0.3,
                "recommended_energy": "measured"
            }

        # Analyze engagement metrics
        total_likes = 0
        total_retweets = 0
        total_replies = 0
        tweet_data = tweets['data']
        tweet_count = len(tweet_data)

        for tweet in tweet_data:
            metrics = tweet.get('public_metrics', {})
            total_likes += metrics.get('like_count', 0)
            total_retweets += metrics.get('retweet_count', 0)
            total_replies += metrics.get('reply_count', 0)

        avg_likes = total_likes / tweet_count if tweet_count > 0 else 0
        avg_retweets = total_retweets / tweet_count if tweet_count > 0 else 0
        avg_replies = total_replies / tweet_count if tweet_count > 0 else 0

        # Calculate engagement rate
        engagement_rate = (total_likes + total_retweets + total_replies) / (tweet_count * 3) if tweet_count > 0 else 0

        # Determine mood and energy based on metrics
        mood = "excited" if engagement_rate > 0.5 else "neutral" if engagement_rate > 0.2 else "subdued"
        energy_level = min(1.0, engagement_rate * 2)
        chaos_factor = min(1.0, (total_replies / (total_likes + 1)) * 2)

        # Determine trend
        trend = "rising" if engagement_rate > 0.5 else "stable" if engagement_rate > 0.2 else "falling"

        # Calculate poast opportunity based on time and engagement
        current_hour = datetime.now().hour
        prime_time = 12 <= current_hour <= 20  # Prime posting hours
        poast_opportunity = min(1.0, engagement_rate + (0.3 if prime_time else 0))

        # Recommend energy level
        recommended_energy = "high" if poast_opportunity > 0.7 else "measured" if poast_opportunity > 0.3 else "low"

        return {
            "current_vibe": {
                "mood": mood,
                "energy_level": energy_level,
                "chaos_factor": chaos_factor
            },
            "trend_direction": {
                "trend": trend,
                "avg_likes": avg_likes,
                "avg_retweets": avg_retweets,
                "avg_replies": avg_replies
            },
            "poast_opportunity": poast_opportunity,
            "recommended_energy": recommended_energy
        }
    except Exception as e:
        click.echo(f"Error analyzing timeline: {str(e)}")
        return {
            "current_vibe": {"mood": "neutral", "energy_level": 0.5, "chaos_factor": 0.0},
            "trend_direction": {"trend": "stable"},
            "poast_opportunity": 0.3,
            "recommended_energy": "measured"
        }

def get_active_bits(db) -> List[Dict]:
    """Get currently active conversation bits"""
    try:
        # Query active bits from database
        active_bits = db["bits"].rows_where(
            "status = 'active' AND expires_at > ?",
            [datetime.now().isoformat()]
        )
        return list(active_bits)
    except Exception as e:
        click.echo(f"Error getting active bits: {str(e)}")
        return []

def get_mutual_history(twitter_api, user_id: Optional[int] = None) -> Dict:
    """Get interaction history with mutual followers"""
    try:
        # Get mutual followers
        mutuals = twitter_api.get_mutual_followers(user_id)
        if not mutuals:
            return {}

        # Get recent interactions with mutuals
        mutual_history = {}
        for mutual_id in mutuals:
            # Get recent interactions using search
            tweets = twitter_api.search_recent_tweets(
                f"from:{mutual_id} to:{twitter_api.user_id}",
                max_results=10
            )
            if tweets and 'data' in tweets:
                mutual_history[mutual_id] = {
                    "recent_interactions": len(tweets['data']),
                    "last_interaction": tweets['data'][0]['created_at'] if tweets['data'] else None
                }

        return mutual_history
    except Exception as e:
        click.echo(f"Error getting mutual history: {str(e)}")
        return {}

async def single_iteration():
    """Run a single iteration of the bot"""
    try:
        # Initialize components
        twitter_api = get_twitter_client()
        db = sqlite_utils.Database("bot.db")
        
        # Ensure database tables exist
        init_db()
        
        # Initialize Anthropic client
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        memory = MemorySystem(db=db, persist_directory=str(VECTOR_DB_PATH))
        personality = PersonalitySystem(memory, anthropic_client=anthropic_client)
        vector_store = VectorStore(persist_directory=str(VECTOR_DB_PATH))
        
        # Analyze timeline
        timeline_analysis = analyze_timeline(twitter_api)
        click.echo("\nTimeline Analysis:")
        click.echo(json.dumps(timeline_analysis, indent=2))
        
        # Get context
        context = {
            'relevant_tweets': [],
            'running_bits': get_active_bits(db),
            'mutual_context': {},
            'mood_context': timeline_analysis['current_vibe'],
            'memory_context': memory.get_recent_memories(),
            'timeline_analysis': timeline_analysis
        }
        
        click.echo("\nContext:")
        click.echo(json.dumps(context, indent=2))
        
        # Get personality state
        personality_state = personality.get_current_state()
        
        click.echo("\nPersonality State:")
        click.echo(f"Mode: {json.dumps(personality_state.get('mode', 'neutral'))}")
        click.echo(f"Energy: {personality_state.get('energy', 1.0)}")
        click.echo(f"Dominant Traits: {json.dumps(personality_state.get('dominant_traits', ['curious', 'analytical']))}")

        # Let Claude decide if we should tweet
        should_tweet = personality.should_tweet_now(context)
        
        if should_tweet:
            # Generate and post content
            content = personality.generate_content(context)
            if content:
                click.echo("\nWould post content:")
                click.echo("--------------------")
                click.echo(content)
                click.echo("--------------------")
                # Log instead of posting
                logger.info(f"Generated content: {content}")
                # Commented out actual posting
                # try:
                #     twitter_api.create_tweet(text=content)
                # except Exception as e:
                #     click.echo(f"Error posting tweet: {str(e)}")

        # Engage with timeline more selectively
        if timeline and 'data' in timeline:
            engaged_count = 0  # Track how many tweets we've engaged with
            engagement_cooldown = {}  # Track when we last engaged with each user
            
            for tweet in timeline['data']:
                # Only engage with a few tweets per iteration
                if engaged_count >= 2:  # Reduced from 3 to 2
                    break
                    
                # Get author info
                author_id = tweet.get('author_id')
                if not author_id:
                    continue
                    
                # Check if we recently engaged with this user
                last_engagement = engagement_cooldown.get(author_id, 0)
                if time.time() - last_engagement < 3600:  # 1 hour cooldown
                    continue
                
                # Decide whether to engage based on personality and context
                should_engage = personality.should_engage(tweet, context)
                if should_engage:
                    metrics = tweet.get('public_metrics', {})
                    # Be more selective about which tweets to engage with
                    if (5 <= metrics.get('like_count', 0) <= 100 and
                        metrics.get('reply_count', 0) < 10):
                        try:
                            # Maybe reply
                            if personality_state['energy'] > 0.7:
                                reply = personality.generate_reply(tweet, context)
                                if reply:
                                    click.echo(f"\nWould reply to tweet {tweet['id']}:")
                                    click.echo("--------------------")
                                    click.echo(f"Original: {tweet.get('text', '')}")
                                    click.echo(f"Reply: {reply}")
                                    click.echo("--------------------")
                                    # Log instead of posting
                                    logger.info(f"Generated reply to {tweet['id']}: {reply}")
                                    # Update engagement tracking
                                    engaged_count += 1
                                    engagement_cooldown[author_id] = time.time()
                        except Exception as e:
                            click.echo(f"Error generating reply: {str(e)}")
        
        # Process any pending user analyses
        pending_users = db["analyzed_users"].rows_where("status = 'pending'", limit=5)
        for user in pending_users:
            try:
                user_data = twitter_api.get_user(user['user_id'])
                if user_data and 'data' in user_data:
                    user_info = user_data['data']
                    # Calculate follow score based on bio, tweets, etc.
                    follow_score = personality.calculate_follow_score(user_info)
                    # Update analyzed users table
                    db["analyzed_users"].update(
                        user['user_id'],
                        {
                            "analysis_result": json.dumps(user_info),
                            "follow_score": follow_score,
                            "status": "analyzed",
                            "last_analyzed": datetime.now().isoformat()
                        }
                    )
                    # Maybe follow if score is high enough
                    if follow_score > 0.8:
                        twitter_api.follow_user(user['user_id'])
            except Exception as e:
                click.echo(f"Error analyzing user {user['user_id']}: {str(e)}")
                
    except Exception as e:
        click.echo(f"Error in iteration: {str(e)}")
        if "--debug" in sys.argv:
            traceback.print_exc()

async def continuous_run():
    """Run the bot continuously"""
    click.echo("Starting continuous run mode...")
    
    # Initialize components
    twitter_api = get_twitter_client()
    db = sqlite_utils.Database("bot.db")
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    memory = MemorySystem(db=db, persist_directory=str(VECTOR_DB_PATH))
    personality = PersonalitySystem(memory, anthropic_client=anthropic_client)
    
    while True:
        try:
            # Get recent activity for decision making
            recent_activity = {
                'last_tweet': None,
                'last_reply': None,
                'last_like': None,
                'last_retweet': None,
                'last_quote': None
            }
            
            # Get recent tweets
            tweets = twitter_api.get_users_tweets(max_results=10)
            if tweets and 'data' in tweets:
                for tweet in tweets['data']:
                    if not recent_activity['last_tweet']:
                        recent_activity['last_tweet'] = tweet
                    if tweet.get('referenced_tweets'):
                        for ref in tweet['referenced_tweets']:
                            if ref['type'] == 'replied_to' and not recent_activity['last_reply']:
                                recent_activity['last_reply'] = tweet
                            elif ref['type'] == 'quoted' and not recent_activity['last_quote']:
                                recent_activity['last_quote'] = tweet
                            elif ref['type'] == 'retweeted' and not recent_activity['last_retweet']:
                                recent_activity['last_retweet'] = tweet
            
            # Run single iteration
            await single_iteration()
            
            # Let Claude decide how long to sleep
            sleep_duration = personality.determine_sleep_duration(recent_activity)
            
            click.echo(f"\nSleeping for {sleep_duration} seconds before next iteration...")
            await asyncio.sleep(sleep_duration)
            
        except Exception as e:
            click.echo(f"Error in continuous run: {str(e)}")
            if "--debug" in sys.argv:
                traceback.print_exc()
            # Sleep for 5 minutes on error before retrying
            click.echo("\nError occurred, sleeping for 5 minutes before retry...")
            await asyncio.sleep(300)

def init_db():
    """Initialize database tables"""
    db = sqlite_utils.Database("bot.db")
    
    # Create memories table if it doesn't exist
    if "memories" not in db.table_names():
        db["memories"].create({
            "id": str,
            "timestamp": str,
            "type": str,
            "content": str,
            "context": str,
            "participants": str,
            "embedding": str
        }, pk="id")
        
    # Create bits table if it doesn't exist
    if "bits" not in db.table_names():
        db["bits"].create({
            "id": str,
            "created_at": str,
            "expires_at": str,
            "type": str,
            "status": str,
            "content": str,
            "context": str,
            "participants": str
        }, pk="id")
        
    # Get existing users before dropping the table
    existing_users = []
    if "analyzed_users" in db.table_names():
        existing_users = [row["user_id"] for row in db["analyzed_users"].rows]
        db["analyzed_users"].drop()
        
    # Create fresh analyzed_users table
    db["analyzed_users"].create({
        "user_id": str,
        "last_analyzed": str,
        "analysis_result": str,
        "follow_score": float,
        "status": str,
        "engagement_rate": float,
        "topic_alignment": float,
        "interaction_score": float,
        "last_interaction": str
    }, pk="user_id")
    db["analyzed_users"].create_index(["status"], if_not_exists=True)
    
    # Reinsert existing users with pending status
    for user_id in existing_users:
        db["analyzed_users"].insert({
            "user_id": user_id,
            "last_analyzed": datetime.now().isoformat(),
            "analysis_result": "{}",
            "follow_score": 0.0,
            "status": "pending",
            "engagement_rate": 0.0,
            "topic_alignment": 0.0,
            "interaction_score": 0.0,
            "last_interaction": ""
        })

def main():
    """Main entry point"""
    try:
        # Initialize database
        init_db()
        
        # Parse command line arguments
        if len(sys.argv) < 2:
            click.echo("Please specify a command: run-once, run")
            return
        
        command = sys.argv[1]
        
        if command == "run-once":
            asyncio.run(single_iteration())
        elif command == "run":
            asyncio.run(continuous_run())
        else:
            click.echo(f"Unknown command: {command}")
            click.echo("Available commands: run-once, run")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        if "--debug" in sys.argv:
            traceback.print_exc()

@click.group()
def cli():
    """Bot management CLI"""
    pass

@cli.command()
def run_once():
    """Run one iteration of the bot"""
    init_db()  # Ensure database is initialized
    asyncio.run(single_iteration())

@cli.command()
def run():
    """Run the bot continuously"""
    init_db()  # Ensure database is initialized
    asyncio.run(continuous_run())

if __name__ == "__main__":
    cli()
