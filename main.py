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

# Import configuration
from config import VECTOR_DB_PATH

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
        memory = MemorySystem(db=db, persist_directory=str(VECTOR_DB_PATH))
        personality = PersonalitySystem(memory)
        vector_store = VectorStore(persist_directory=str(VECTOR_DB_PATH))
        
        # Analyze timeline
        timeline_analysis = analyze_timeline(twitter_api)
        click.echo("\nTimeline Analysis:")
        click.echo(json.dumps(timeline_analysis, indent=2))
        
        # Get context
        context = {
            'relevant_tweets': [],  # Will be populated by vector search
            'running_bits': get_active_bits(db),
            'mutual_context': {},  # Removed mutual follower context
            'mood_context': timeline_analysis['current_vibe'],
            'memory_context': memory.get_recent_memories()  # Now properly getting memories
        }
        
        click.echo("\nContext:")
        click.echo(json.dumps(context, indent=2))
        
        # Get personality state
        personality_state = personality.get_current_state()
        
        click.echo("\nPersonality State:")
        click.echo(f"Mode: {json.dumps(personality_state.get('mode', 'neutral'))}")
        click.echo(f"Energy: {personality_state.get('energy', 1.0)}")
        click.echo(f"Dominant Traits: {json.dumps(personality_state.get('dominant_traits', ['curious', 'analytical']))}")
        
    except Exception as e:
        click.echo(f"Error in iteration: {str(e)}")
        if "--debug" in sys.argv:
            traceback.print_exc()

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

def main():
    """Main entry point"""
    try:
        # Initialize database
        init_db()
        
        # Parse command line arguments
        if len(sys.argv) < 2:
            click.echo("Please specify a command: run-once")
            return
        
        command = sys.argv[1]
        
        if command == "run-once":
            asyncio.run(single_iteration())
        else:
            click.echo(f"Unknown command: {command}")
            
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
    import asyncio
    asyncio.run(single_iteration())

if __name__ == "__main__":
    main()
