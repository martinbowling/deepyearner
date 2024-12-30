# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "sqlite-utils",
#     "tweepy",
#     "anthropic",
#     "chromadb",
#     "openai",
#     "numpy",
#     "pickle",
#     "tqdm"
# ]
# ///

"""
SINGLE-FILE CLAUDE-STYLE TWITTER BOT WITH RAG (ChromaDB)
--------------------------------------------------------
Features:
1. SQLite DB for storing tweets and bot states
2. Tweepy for Twitter integration
3. Anthropic for LLM (Claude-3.5-Sonnet)
4. ChromaDB for storing tweet embeddings and retrieving relevant context
5. Random pause between loop iterations

Usage:
    python bot_rag.py setup_db
    python bot_rag.py run_once
    python bot_rag.py run_forever
"""

import click
import sqlite_utils
import tweepy
import anthropic
import json
import time
import asyncio
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from openai import OpenAI
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Set up logging
logger = logging.getLogger(__name__)

# ChromaDB imports
import chromadb
from chromadb.config import Settings

###############################################################################
# 0. Configuration
###############################################################################

DB_PATH = "twitter_bot.db"
PROMPTS_DIR = Path(__file__).parent / "prompts"
VECTOR_DB_PATH = Path(__file__).parent / "vector_db"

# Twitter credentials from environment variables
CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

from config import ModelType, get_model_name, ModelConfig

class Bot:
    def __init__(self, config: Dict[str, Any]):
        """Initialize bot with model configurations"""
        self.config = config
        self.chat_model = get_model_name(ModelType.CHAT)
        self.embedding_model = get_model_name(ModelType.EMBEDDING)

###############################################################################
# 1. Personality State Management
###############################################################################

@dataclass
class PersonalityMode:
    intellectual: float = 0.0
    poasting: float = 0.0
    yearning: float = 0.0
    blessed: float = 0.0
    cursed: float = 0.0

class PersonalityState:
    def __init__(self):
        self.current_mode = PersonalityMode()
        self.energy_history: List[Dict] = []
        self.recent_vibes: List[Dict] = []
        self.last_transition = datetime.now()
        
    def update_state(self, timeline_state: Dict, interaction_results: Dict) -> None:
        """Update personality metrics based on recent actions"""
        # Update energy history
        self.energy_history.append({
            "timestamp": datetime.now(),
            "timeline_mood": timeline_state.get("mood", "neutral"),
            "interaction_success": interaction_results.get("success_rate", 0.5)
        })
        
        # Keep history manageable
        if len(self.energy_history) > 100:
            self.energy_history = self.energy_history[-100:]
            
        # Update recent vibes
        self.recent_vibes.append(timeline_state)
        if len(self.recent_vibes) > 10:
            self.recent_vibes = self.recent_vibes[-10:]

    def should_transition(self) -> bool:
        """Determine if we should shift modes"""
        time_since_last = (datetime.now() - self.last_transition).total_seconds()
        if time_since_last < 3600:  # Don't transition more than once per hour
            return False
            
        # Analyze recent history to determine if transition needed
        recent_mood = self.analyze_recent_mood()
        current_dominant = self.get_dominant_mode()
        
        return recent_mood != current_dominant

    def analyze_recent_mood(self) -> str:
        """Analyze recent interactions to determine mood"""
        if not self.recent_vibes:
            return "neutral"
            
        mood_counts = {}
        for vibe in self.recent_vibes:
            mood = vibe.get("mood", "neutral")
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
        return max(mood_counts.items(), key=lambda x: x[1])[0]

    def get_dominant_mode(self) -> str:
        """Get current dominant personality mode"""
        modes = {
            "intellectual": self.current_mode.intellectual,
            "poasting": self.current_mode.poasting,
            "yearning": self.current_mode.yearning,
            "blessed": self.current_mode.blessed,
            "cursed": self.current_mode.cursed
        }
        return max(modes.items(), key=lambda x: x[1])[0]

###############################################################################
# 2. Vibe Analysis
###############################################################################

class VibeAnalyzer:
    def __init__(self):
        self.recent_moods: List[Dict] = []
        self.trend_memory: Dict[str, List[float]] = {}
        
    def analyze_timeline(self, tweets: List[Dict], personality_state: PersonalityState) -> Dict:
        current_vibe = self.detect_mood(tweets)
        self.recent_moods.append(current_vibe)
        
        if len(self.recent_moods) > 20:
            self.recent_moods = self.recent_moods[-20:]
            
        return {
            "current_vibe": current_vibe,
            "trend_direction": self.track_trends(),
            "poast_opportunity": self.calculate_poast_potential(),
            "recommended_energy": self.suggest_energy_level(personality_state)
        }
        
    def detect_mood(self, tweets: List[Dict]) -> Dict:
        """Analyze tweets to detect current timeline mood"""
        # This would use the LLM to analyze tweet content
        return {
            "mood": "neutral",
            "energy_level": 0.5,
            "chaos_factor": 0.0
        }
        
    def track_trends(self) -> Dict:
        """Track mood and engagement trends"""
        if not self.recent_moods:
            return {"trend": "stable"}
            
        # Simple trend analysis
        recent_energy = [m.get("energy_level", 0.5) for m in self.recent_moods[-5:]]
        avg_energy = sum(recent_energy) / len(recent_energy)
        
        if avg_energy > 0.7:
            return {"trend": "rising"}
        elif avg_energy < 0.3:
            return {"trend": "falling"}
        return {"trend": "stable"}
        
    def calculate_poast_potential(self) -> float:
        """Calculate potential for successful poasting"""
        if not self.recent_moods:
            return 0.5
            
        recent_chaos = [m.get("chaos_factor", 0.0) for m in self.recent_moods[-5:]]
        avg_chaos = sum(recent_chaos) / len(recent_chaos)
        
        # Higher chaos = higher poasting potential
        return min(1.0, avg_chaos + 0.3)
        
    def suggest_energy_level(self, personality_state: PersonalityState) -> str:
        """Suggest appropriate energy level based on personality state"""
        dominant_mode = personality_state.get_dominant_mode()
        
        if dominant_mode in ["blessed", "cursed"]:
            return "high"
        elif dominant_mode == "intellectual":
            return "measured"
        return "adaptive"

###############################################################################
# 3. Prompt Management
###############################################################################

class PromptManager:
    def __init__(self):
        self.prompts = self.load_all_prompts()
        self.recent_successes: List[Dict] = []
        
    def load_all_prompts(self) -> Dict[str, str]:
        """Load all prompt templates from the prompts directory"""
        prompts = {}
        for prompt_file in PROMPTS_DIR.glob("*.txt"):
            prompt_name = prompt_file.stem
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts[prompt_name] = f.read()
        return prompts
        
    def get_contextualized_prompt(
        self, 
        prompt_type: str, 
        personality_state: PersonalityState, 
        rag_context: Dict
    ) -> str:
        """Get a prompt enhanced with current context"""
        base_prompt = self.prompts.get(prompt_type)
        if not base_prompt:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        return self.enhance_prompt(base_prompt, personality_state, rag_context)
        
    def enhance_prompt(
        self, 
        base_prompt: str, 
        personality_state: PersonalityState, 
        rag_context: Dict
    ) -> str:
        """Enhance a base prompt with personality and RAG context"""
        # Convert personality state to JSON for template
        personality_json = json.dumps({
            "current_mode": personality_state.current_mode.__dict__,
            "recent_vibes": personality_state.recent_vibes[-3:]  # Last 3 vibes
        }, indent=2)
        
        # Format RAG context
        rag_json = json.dumps({
            "relevant_tweets": rag_context.get("relevant_tweets", [])[:3],
            "running_bits": rag_context.get("running_bits", [])[:2],
            "mutual_context": rag_context.get("mutual_context", {})
        }, indent=2)
        
        # Replace placeholders in the prompt template
        enhanced = base_prompt.replace("{{PERSONALITY_STATE}}", personality_json)
        enhanced = enhanced.replace("{{RAG_CONTEXT}}", rag_json)
        
        return enhanced

###############################################################################
# 4. Database Setup
###############################################################################

def init_db():
    db = sqlite_utils.Database(DB_PATH)
    
    # Core tables
    if "tweets" not in db.table_names():
        db["tweets"].create(
            {
                "id": str,
                "text": str,
                "created_at": str,
                "author_id": str,
                "prompt_type": str,
                "raw_response": str
            },
            pk="id"
        )
    
    if "interactions" not in db.table_names():
        db["interactions"].create(
            {
                "id": int,
                "tweet_id": str,
                "user_id": str,
                "interaction_type": str,
                "timestamp": str,
                "raw_response": str
            },
            pk="id"
        )
    
    if "bot_states" not in db.table_names():
        db["bot_states"].create(
            {
                "id": int,
                "state_key": str,
                "state_json": str
            },
            pk="id"
        )
        
    # New tables for enhanced functionality
    if "personality_states" not in db.table_names():
        db["personality_states"].create(
            {
                "id": int,
                "timestamp": str,
                "mode": str,
                "energy_levels": str,
                "active_bits": str
            },
            pk="id"
        )
    
    if "meme_tracker" not in db.table_names():
        db["meme_tracker"].create(
            {
                "id": int,
                "meme_type": str,
                "first_seen": str,
                "last_used": str,
                "success_rate": float
            },
            pk="id"
        )
    
    if "poast_metrics" not in db.table_names():
        db["poast_metrics"].create(
            {
                "id": int,
                "tweet_id": str,
                "vibe_score": float,
                "chaos_level": float,
                "intellectual_depth": float,
                "meme_relevance": float
            },
            pk="id"
        )
        
    # Follow management tables
    if "follow_decisions" not in db.table_names():
        db["follow_decisions"].create({
            "id": int,
            "username": str,
            "decision": bool,
            "confidence": float,
            "reasoning": str,
            "follow_type": str,
            "context_type": str,
            "review_after": str,
            "created_at": str,
            "reviewed_at": str
        }, pk="id")
        
    if "unfollow_log" not in db.table_names():
        db["unfollow_log"].create({
            "id": int,
            "username": str,
            "reason": str,
            "unfollowed_at": str
        }, pk="id")
    
    return db

###############################################################################
# 5. Enhanced Vector Store
###############################################################################

class ContextualVectorStore:
    def __init__(self, name: str = "twitter_vectors"):
        self.name = name
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict] = []
        self.query_cache: Dict[str, List[float]] = {}
        self.db_path = VECTOR_DB_PATH / f"{name}.pkl"
        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()
        
    def situate_context(self, tweet_context: Dict, tweet: Dict) -> Tuple[str, Any]:
        """Generate contextual information for a tweet within its social context
        
        Args:
            tweet_context: Dict containing contextual info like replies, quoted tweets, etc.
            tweet: The main tweet to contextualize
        """
        TWEET_CONTEXT_PROMPT = """
        Analyze this tweet's context:
        
        Author: {author}
        Timestamp: {created_at}
        
        Previous context:
        {previous_tweets}
        
        Replies:
        {replies}
        
        Quoted/Referenced tweets:
        {quoted_tweets}
        
        Recent hashtags and topics:
        {recent_topics}
        """

        TWEET_ANALYSIS_PROMPT = """
        Here is the tweet we want to analyze for search retrieval:
        <tweet>
        {tweet_text}
        </tweet>

        Please provide a concise contextual summary that captures:
        1. The key topics and themes
        2. The tweet's relationship to ongoing conversations
        3. Any relevant memes or cultural references
        4. The general mood/tone
        
        Keep the summary brief and focused on improving search relevance.
        Answer only with the contextual summary and nothing else.
        """

        # Extract context fields with defaults
        context = {
            'author': tweet_context.get('author', 'Unknown'),
            'created_at': tweet_context.get('created_at', 'Unknown time'),
            'previous_tweets': '\n'.join(tweet_context.get('previous_tweets', ['No previous context'])),
            'replies': '\n'.join(tweet_context.get('replies', ['No replies'])),
            'quoted_tweets': '\n'.join(tweet_context.get('quoted_tweets', ['No quoted tweets'])),
            'recent_topics': '\n'.join(tweet_context.get('recent_topics', ['No recent topics']))
        }

        response = anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": TWEET_CONTEXT_PROMPT.format(**context),
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": TWEET_ANALYSIS_PROMPT.format(tweet_text=tweet.get('text', '')),
                        }
                    ]
                }
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        return response.content[0].text, response.usage

    def process_tweet_batch(
        self, 
        tweets: List[Dict],
        parallel_threads: int = 4
    ) -> None:
        """Process a batch of tweets with parallel contextualization"""
        
        def process_tweet(tweet: Dict) -> Dict:
            # Get tweet context
            tweet_context = {
                'author': tweet.get('author', {}).get('username', 'unknown'),
                'created_at': tweet.get('created_at', 'unknown time'),
                'previous_tweets': tweet.get('conversation_context', []),
                'replies': tweet.get('replies', []),
                'quoted_tweets': tweet.get('quoted_tweets', []),
                'recent_topics': tweet.get('recent_topics', [])
            }
            
            contextualized_text, usage = self.situate_context(tweet_context, tweet)
            
            with self.token_lock:
                self.token_counts['input'] += usage.input_tokens
                self.token_counts['output'] += usage.output_tokens
                self.token_counts['cache_read'] += usage.cache_read_input_tokens
                self.token_counts['cache_creation'] += usage.cache_creation_input_tokens
            
            return {
                'text_to_embed': f"{tweet['text']}\n\n{contextualized_text}",
                'metadata': {
                    'tweet_id': tweet['id'],
                    'created_at': tweet.get('created_at'),
                    'author': tweet.get('author', {}).get('username', 'unknown'),
                    'original_text': tweet['text'],
                    'context': contextualized_text
                }
            }

        print(f"Processing {len(tweets)} tweets with {parallel_threads} threads")
        texts_to_embed = []
        metadata = []
        
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for tweet in tweets:
                futures.append(executor.submit(process_tweet, tweet))
            
            for future in tqdm(as_completed(futures), total=len(tweets), desc="Processing tweets"):
                result = future.result()
                texts_to_embed.append(result['text_to_embed'])
                metadata.append(result['metadata'])

        # Create embeddings in batches
        batch_size = 128
        all_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            batch_embeddings = [self.create_embedding(text) for text in batch]
            all_embeddings.extend(batch_embeddings)

        # Update store
        self.embeddings.extend(all_embeddings)
        self.metadata.extend(metadata)
        self.save_db()

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using OpenAI's text-embedding-3-small"""
        response = openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def search(
        self, 
        query: str, 
        k: int = 20,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar tweets"""
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.create_embedding(query)
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector store")

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        
        # Filter and return results
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity < min_similarity:
                break
            if len(results) >= k:
                break
                
            results.append({
                "metadata": self.metadata[idx],
                "similarity": similarity
            })
            
        return results

    def save_db(self) -> None:
        """Save vector store to disk"""
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": self.query_cache,
            "token_counts": self.token_counts
        }
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)

    def load_db(self) -> None:
        """Load vector store from disk"""
        if not self.db_path.exists():
            return
            
        with open(self.db_path, "rb") as f:
            data = pickle.dump(f)
            self.embeddings = data["embeddings"]
            self.metadata = data["metadata"]
            self.query_cache = data["query_cache"]
            self.token_counts = data["token_counts"]

###############################################################################
# 6. Enhanced RAG Implementation
###############################################################################

class EnhancedRAG:
    def __init__(self):
        self.vector_store = ContextualVectorStore()
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(VECTOR_DB_PATH / "chroma")
        ))
        
    def get_contextual_memory(
        self,
        query: str,
        personality_state: 'PersonalityState',
        db: sqlite_utils.Database,
        k: int = 5,
        memory_context: Dict = {}
    ) -> Dict:
        """Get enhanced contextual memory using both vector and traditional search"""
        
        # Get relevant tweets from vector store
        vector_results = self.vector_store.search(query, k=k)
        
        # Get active memes and bits
        active_bits = self._get_active_bits(db)
        
        # Get mutual history
        mutual_context = self._get_mutual_history(db)
        
        # Combine results
        return {
            "relevant_tweets": vector_results,
            "running_bits": active_bits,
            "mutual_context": mutual_context,
            "mood_context": personality_state.recent_vibes,
            "memory_context": memory_context
        }
        
    def ingest_tweets(self, tweets: List[Dict]) -> None:
        """Ingest new tweets into both vector store and ChromaDB"""
        # Process with vector store
        self.vector_store.process_tweet_batch(tweets)
        
        # Also store in ChromaDB for backup/comparison
        collection = self._get_or_create_collection()
        
        documents = []
        metadatas = []
        ids = []
        
        for tweet in tweets:
            documents.append(tweet['text'])
            metadatas.append({
                'tweet_id': str(tweet['id']),
                'created_at': tweet['created_at'],
                'author': tweet.get('author', 'unknown')
            })
            ids.append(str(tweet['id']))
            
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        collection_name = "tweets"
        try:
            return self.chroma_client.get_collection(collection_name)
        except ValueError:
            return self.chroma_client.create_collection(collection_name)
            
    def _get_active_bits(self, db: sqlite_utils.Database) -> List[Dict]:
        """Get currently active memes and bits"""
        return get_active_bits(db)  # Reuse existing function
        
    def _get_mutual_history(self, db: sqlite_utils.Database) -> Dict:
        """Get mutual interaction history"""
        return get_mutual_history(db)  # Reuse existing function

###############################################################################
# 7. Database Setup
###############################################################################

def init_db():
    db = sqlite_utils.Database(DB_PATH)
    
    # Core tables
    if "tweets" not in db.table_names():
        db["tweets"].create(
            {
                "id": str,
                "text": str,
                "created_at": str,
                "author_id": str,
                "prompt_type": str,
                "raw_response": str
            },
            pk="id"
        )
    
    if "interactions" not in db.table_names():
        db["interactions"].create(
            {
                "id": int,
                "tweet_id": str,
                "user_id": str,
                "interaction_type": str,
                "timestamp": str,
                "raw_response": str
            },
            pk="id"
        )
    
    if "bot_states" not in db.table_names():
        db["bot_states"].create(
            {
                "id": int,
                "state_key": str,
                "state_json": str
            },
            pk="id"
        )
        
    # New tables for enhanced functionality
    if "personality_states" not in db.table_names():
        db["personality_states"].create(
            {
                "id": int,
                "timestamp": str,
                "mode": str,
                "energy_levels": str,
                "active_bits": str
            },
            pk="id"
        )
    
    if "meme_tracker" not in db.table_names():
        db["meme_tracker"].create(
            {
                "id": int,
                "meme_type": str,
                "first_seen": str,
                "last_used": str,
                "success_rate": float
            },
            pk="id"
        )
    
    if "poast_metrics" not in db.table_names():
        db["poast_metrics"].create(
            {
                "id": int,
                "tweet_id": str,
                "vibe_score": float,
                "chaos_level": float,
                "intellectual_depth": float,
                "meme_relevance": float
            },
            pk="id"
        )
        
    # Follow management tables
    if "follow_decisions" not in db.table_names():
        db["follow_decisions"].create({
            "id": int,
            "username": str,
            "decision": bool,
            "confidence": float,
            "reasoning": str,
            "follow_type": str,
            "context_type": str,
            "review_after": str,
            "created_at": str,
            "reviewed_at": str
        }, pk="id")
        
    if "unfollow_log" not in db.table_names():
        db["unfollow_log"].create({
            "id": int,
            "username": str,
            "reason": str,
            "unfollowed_at": str
        }, pk="id")
    
    return db

###############################################################################
# 8. Twitter Integration
###############################################################################

def get_twitter_api():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api

def post_tweet(text: str) -> str:
    api = get_twitter_api()
    resp = api.update_status(status=text)
    return str(resp.id)

def reply_to_tweet(tweet_id: str, text: str) -> str:
    api = get_twitter_api()
    resp = api.update_status(
        status=text,
        in_reply_to_status_id=int(tweet_id),
        auto_populate_reply_metadata=True
    )
    return str(resp.id)

###############################################################################
# 9. Follow Management
###############################################################################

class FollowManager:
    def __init__(self, db: sqlite_utils.Database, twitter_api):
        self.db = db
        self.twitter_api = twitter_api
        self.prompt_manager = PromptManager()
        
    async def evaluate_account_for_follow(
        self, 
        username: str,
        context_type: str = "organic",
        max_tweets: int = 10
    ) -> Dict:
        """Evaluate whether to follow an account based on various factors"""
        try:
            # Get account info
            user = self.twitter_api.get_user(screen_name=username)
            
            # Get recent tweets
            recent_tweets = self.twitter_api.user_timeline(
                screen_name=username,
                count=max_tweets,
                tweet_mode="extended"
            )
            
            # Get mutual followers
            mutuals = self._get_mutual_followers(username)
            
            # Get recent interactions
            interactions = self._get_recent_interactions(username)
            
            # Prepare context for prompt
            context = {
                "username": username,
                "display_name": user.name,
                "bio": user.description,
                "follower_count": user.followers_count,
                "following_count": user.friends_count,
                "tweet_count": user.statuses_count,
                "account_age_days": (datetime.now() - user.created_at).days,
                "is_verified": user.verified,
                "recent_tweets": "\n".join([t.full_text for t in recent_tweets]),
                "mutual_followers": json.dumps(mutuals, indent=2),
                "recent_interactions": json.dumps(interactions, indent=2)
            }
            
            # Get decision from Claude
            decision = await self._get_follow_decision(context)
            
            # Store decision for tracking
            self._store_follow_decision(username, decision, context_type)
            
            return decision
            
        except Exception as e:
            print(f"Error evaluating account {username}: {str(e)}")
            return None
            
    async def review_following(self, review_count: int = 10) -> None:
        """Randomly review a subset of followed accounts"""
        following = self.db["follow_decisions"].rows_where(
            "decision = ? AND reviewed_at < datetime('now', '-7 days')",
            [True],
            order_by="RANDOM()",
            limit=review_count
        )
        
        for account in following:
            decision = await self.evaluate_account_for_follow(
                account["username"],
                context_type="review"
            )
            
            if decision and not decision["follow_decision"]:
                # Unfollow if conditions are met
                self._unfollow_account(account["username"], decision["reasoning"])
                
    def _get_mutual_followers(self, username: str) -> List[Dict]:
        """Get information about mutual followers"""
        try:
            # Get mutual followers
            user_followers = set(self.twitter_api.followers_ids(screen_name=username))
            my_followers = set(self.twitter_api.followers_ids())
            mutuals = user_followers.intersection(my_followers)
            
            # Get detailed info for up to 10 random mutuals
            mutual_details = []
            for user_id in random.sample(mutuals, min(10, len(mutuals))):
                user = self.twitter_api.get_user(user_id=user_id)
                mutual_details.append({
                    "username": user.screen_name,
                    "display_name": user.name,
                    "followers": user.followers_count
                })
                
            return mutual_details
            
        except Exception as e:
            print(f"Error getting mutuals for {username}: {str(e)}")
            return []
            
    def _get_recent_interactions(self, username: str) -> List[Dict]:
        """Get recent interactions with the account"""
        return list(self.db["interactions"].rows_where(
            "user_id = ? AND timestamp > datetime('now', '-30 days')",
            [username],
            order_by="timestamp desc",
            limit=10
        ))
        
    async def _get_follow_decision(self, context: Dict) -> Dict:
        """Get follow decision from Claude using prompt"""
        prompt = self.prompt_manager.load_prompt("follow_decision.txt")
        formatted_prompt = prompt.format(**context)
        
        response = await anthropic_client.messages.create(
            model=self.chat_model,
            max_tokens=1000,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": formatted_prompt
            }]
        )
        
        # Extract decision JSON from response
        decision_text = response.content[0].text
        decision_match = re.search(r'<decision>(.*?)</decision>', 
                                 decision_text, 
                                 re.DOTALL)
                                 
        if decision_match:
            try:
                return json.loads(decision_match.group(1))
            except json.JSONDecodeError:
                print("Error parsing decision JSON")
                return None
        return None
        
    def _store_follow_decision(
        self, 
        username: str, 
        decision: Dict,
        context_type: str
    ) -> None:
        """Store follow decision in database"""
        self.db["follow_decisions"].insert({
            "username": username,
            "decision": decision["follow_decision"],
            "confidence": decision["confidence_score"],
            "reasoning": json.dumps(decision["reasoning"]),
            "follow_type": decision["follow_type"],
            "context_type": context_type,
            "review_after": datetime.now().strftime("%Y-%m-%d"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    def _unfollow_account(self, username: str, reason: Dict) -> None:
        """Unfollow an account and log the reason"""
        try:
            self.twitter_api.destroy_friendship(screen_name=username)
            
            self.db["unfollow_log"].insert({
                "username": username,
                "reason": json.dumps(reason),
                "unfollowed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            print(f"Error unfollowing {username}: {str(e)}")

###############################################################################
# 10. Enhanced Agent Loop
###############################################################################

from reactive_system import ReactiveSystem
from content_discovery import ContentDiscovery
from personality_system import PersonalitySystem, PersonalityMode, MoodType
from memory_system import MemorySystem
from content_generator import ContentGenerator
from researcher import ResearchManager
from twitter_actions import TwitterActions

async def enhanced_agentic_loop():
    """
    Enhanced agent loop with advanced personality management, improved RAG,
    reactive engagement, content discovery, and enhanced memory
    """
    # Initialize components
    db = init_db()
    rag = EnhancedRAG()
    memory_system = MemorySystem(db, rag.vector_store)
    personality = PersonalitySystem(db)
    vibe_analyzer = VibeAnalyzer()
    prompt_manager = PromptManager()
    twitter_api = get_twitter_api()
    twitter_actions = TwitterActions(twitter_api)
    content_generator = ContentGenerator(
        memory_system=memory_system,
        personality=personality,
        anthropic_client=get_anthropic_client()
    )
    researcher = ResearchManager(
        memory_system=memory_system,
        content_fetcher=ContentFetcher(
            markdowner_api_key=os.getenv("MARKDOWNER_API_KEY"),
            memory_system=memory_system,
            anthropic_client=get_anthropic_client()
        ),
        personality=personality,
        anthropic_client=get_anthropic_client(),
        search_config=SearchConfig(
            api_key=os.getenv("SCALESERP_API_KEY"),
            max_results=10
        )
    )
    
    # Initialize reactive and discovery systems
    reactive_system = ReactiveSystem(
        twitter_api=twitter_api,
        personality=personality,
        prompt_manager=prompt_manager,
        db=db,
        vector_store=rag.vector_store,
        memory_system=memory_system
    )
    
    content_discovery = ContentDiscovery(
        twitter_api=twitter_api,
        personality=personality,
        prompt_manager=prompt_manager,
        db=db,
        vector_store=rag.vector_store,
        memory_system=memory_system
    )
    
    while True:
        try:
            # 1. Process reactive engagements
            await reactive_system.process_new_interactions()
            
            # 2. Discover content opportunities
            discoveries = await content_discovery.discover_content()
            
            # 3. Get timeline data
            api = get_twitter_api()
            public_tweets = api.home_timeline(count=10)
            tweet_data = [{
                "id": str(t.id),
                "text": t.text,
                "author": t.user.screen_name,
                "created_at": str(t.created_at)
            } for t in public_tweets]
            
            # 4. Analyze timeline vibe
            timeline_analysis = vibe_analyzer.analyze_timeline(tweet_data, personality)
            
            # 5. Get memory context
            memory_context = {
                'recent_episodes': memory_system.get_relevant_episodes(
                    "\n".join(t["text"] for t in tweet_data),
                    limit=5
                ),
                'active_relationships': [
                    memory_system.get_relationship(t['author'])
                    for t in tweet_data
                    if memory_system.get_relationship(t['author'])
                ],
                'working_memory': memory_system.working_memory
            }
            
            # 6. Get RAG context with memory
            rag_context = rag.get_contextual_memory(
                "\n".join(t["text"] for t in tweet_data),
                personality,
                db,
                memory_context
            )
            
            # 7. Update personality state
            personality.update_from_timeline(timeline_analysis)
            personality.update_from_discoveries(discoveries)
            
            # Get recent interactions from memory
            recent_interactions = memory_system.get_relevant_episodes(
                type="interaction",
                limit=10
            )
            personality.update_from_interactions([
                e.content for e in recent_interactions
            ])
            
            # Update any ongoing transition
            personality.update_transition()
            
            # 8. Memory consolidation
            memory_system.consolidate_memory()
            
            # Get memory insights if needed
            if random.random() < 0.1:  # 10% chance each cycle
                consolidation_prompt = prompt_manager.get_contextualized_prompt(
                    "memory_consolidation",
                    personality,
                    {
                        'recent_episodes': recent_interactions,
                        'active_relationships': memory_context['active_relationships'],
                        'key_concepts': memory_system.get_related_concepts("twitter", 0.3),
                        'recent_connections': [
                            r for r in memory_system.relationships.values()
                            if r.last_interaction and
                            datetime.now() - r.last_interaction < timedelta(days=7)
                        ],
                        'working_memory': memory_context['working_memory'],
                        'attention_weights': memory_system.attention_weights
                    }
                )
                
                insights_response = call_anthropic_api(consolidation_prompt)
                insights = extract_json_section(insights_response)
                
                # Store insights in memory
                memory_system.add_episode(
                    type="insights",
                    content=insights,
                    context={
                        'source': 'memory_consolidation',
                        'timestamp': datetime.now().isoformat()
                    },
                    participants=set()
                )
            
            # 9. Check if personality should transition
            if personality.should_transition():
                transition_prompt = prompt_manager.get_contextualized_prompt(
                    "personality_transition",
                    personality,
                    {
                        'current_mode': personality.current_mode.to_dict(),
                        'energy_levels': personality.energy.get_state_summary(),
                        'dominant_traits': [
                            t.value for t in personality.current_mode.get_dominant_traits()
                        ],
                        'last_transition': personality.last_transition.isoformat(),
                        'timeline_vibe': timeline_analysis,
                        'engagement_level': timeline_analysis.get('engagement_level', 0.5),
                        'key_topics': timeline_analysis.get('key_topics', []),
                        'recent_interactions': [e.content for e in recent_interactions],
                        'recent_memories': memory_system.get_relevant_episodes(limit=5)
                    }
                )
                
                transition_response = call_anthropic_api(transition_prompt)
                transition_data = extract_json_section(transition_response)
                
                if transition_data.get('should_transition', False):
                    target_mode = PersonalityMode.from_dict(
                        transition_data['recommended_mode']
                    )
                    
                    personality.start_transition(target_mode)
                    
                    # Store transition in memory
                    memory_system.add_episode(
                        type="transition",
                        content={
                            'from_mode': personality.current_mode.to_dict(),
                            'to_mode': target_mode.to_dict(),
                            'analysis': transition_data['analysis'],
                            'strategy': transition_data['transition_strategy']
                        },
                        context={
                            'timeline_analysis': timeline_analysis,
                            'memory_context': memory_context
                        },
                        participants=set()
                    )
            
            # 10. Process discoveries
            if discoveries:
                for discovery in discoveries:
                    if discovery['confidence'] > 0.8:
                        # Store discovery in memory
                        memory_system.add_episode(
                            type=f"discovery_{discovery['type']}",
                            content=discovery,
                            context={
                                'timeline_analysis': timeline_analysis,
                                'personality_state': personality.get_state_summary()
                            },
                            participants={
                                p for p in discovery.get('participants', [])
                            }
                        )
                        
                        # Handle discovery
                        if discovery['type'] == 'trend':
                            await handle_trend_discovery(
                                discovery,
                                personality,
                                twitter_api,
                                db,
                                memory_system
                            )
                        elif discovery['type'] == 'conversation':
                            await handle_conversation_discovery(
                                discovery,
                                personality,
                                twitter_api,
                                db,
                                memory_system
                            )
                        elif discovery['type'] == 'search_result':
                            await handle_search_discovery(
                                discovery,
                                personality,
                                twitter_api,
                                db,
                                memory_system
                            )
            
            # 11. Check if rest needed
            if personality.energy.should_rest():
                logger.info("Taking a rest to recover energy...")
                personality.energy.take_rest()
                
                # Store rest episode
                memory_system.add_episode(
                    type="rest",
                    content={
                        'duration_minutes': random.randint(5, 10),
                        'energy_before': personality.energy.get_state_summary(),
                        'reason': 'energy_depletion'
                    },
                    context={
                        'personality_state': personality.get_state_summary(),
                        'timeline_analysis': timeline_analysis
                    },
                    participants=set()
                )
                
                await asyncio.sleep(random.randint(300, 600))
                continue
            
            # 12. Check for research opportunities
            research_session = await researcher.manage_research()
            if research_session:
                logger.info(
                    f"Completed research on {research_session.topic} "
                    f"with quality {research_session.research_quality}"
                )
                
                # Consider tweeting insights
                if (
                    research_session.research_quality > 0.7 and
                    random.random() < 0.3  # 30% chance
                ):
                    # Generate insight thread
                    tweets = await content_generator.generate_thread(
                        main_topic=research_session.topic,
                        relevant_content=research_session.content_found,
                        max_tweets=5
                    )
                    
                    if tweets:
                        # Post thread
                        thread_ids = twitter_actions.post_thread(tweets)
                        
                        if thread_ids:
                            # Store thread in memory
                            memory_system.add_episode(
                                type="research_thread",
                                content={
                                    'topic': research_session.topic,
                                    'tweet_ids': thread_ids,
                                    'tweets': tweets,
                                    'research_quality': research_session.research_quality,
                                    'insights': research_session.insights_gained
                                },
                                context={
                                    'research_session': {
                                        'duration_minutes': (
                                            research_session.end_time -
                                            research_session.start_time
                                        ).total_seconds() / 60,
                                        'content_found': len(research_session.content_found),
                                        'new_topics': research_session.new_topics_discovered
                                    }
                                },
                                participants=set()
                            )
            
            # 13. Decide on additional proactive action
            monitor_prompt = prompt_manager.get_contextualized_prompt(
                "timeline_monitor",
                personality,
                {**rag_context, 'memory_context': memory_context}
            )
            
            monitor_response = call_anthropic_api(monitor_prompt)
            monitor_state = extract_json_section(monitor_response)
            
            # 14. Take proactive action if needed and energy permits
            should_post = (
                monitor_state.get("action_needed", False) and
                personality.energy.creative_energy > 0.4
            )
            
            if should_post:
                # Get content suggestions
                suggestions = await content_generator.generate_content_suggestions(
                    current_context={
                        'timeline_analysis': timeline_analysis,
                        'topics': timeline_analysis.get('key_topics', []),
                        'time_of_day': datetime.now().hour,
                        'recent_interactions': [
                            e.content for e in recent_interactions
                        ],
                        'memory_context': memory_context
                    },
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
                    
                    # Post tweet
                    new_tweet_id = twitter_actions.post_tweet(
                        best_suggestion.tweet_text
                    )
                    
                    if new_tweet_id:
                        # Update energy based on content type
                        intensity = {
                            'original': 0.3,
                            'insight': 0.4,
                            'commentary': 0.2
                        }.get(best_suggestion.content_type, 0.3)
                        
                        personality.energy.update_energy(
                            'creative',
                            intensity
                        )
                        
                        # Store in memory
                        memory_system.add_episode(
                            type="generated_content",
                            content={
                                'tweet_id': new_tweet_id,
                                'text': best_suggestion.tweet_text,
                                'source_urls': best_suggestion.source_urls,
                                'topics': best_suggestion.topics,
                                'content_type': best_suggestion.content_type,
                                'reference_content': best_suggestion.reference_content,
                                'personality_state': personality.get_state_summary()
                            },
                            context={
                                'timeline_analysis': timeline_analysis,
                                'memory_context': memory_context,
                                'generation_context': {
                                    'confidence': best_suggestion.confidence,
                                    'personality_alignment': best_suggestion.personality_alignment,
                                    'timing_score': best_suggestion.timing_score,
                                    'expected_engagement': best_suggestion.expected_engagement
                                }
                            },
                            participants=set()
                        )
                        
                        # Store in DB
                        db["tweets"].insert({
                            "id": new_tweet_id,
                            "text": best_suggestion.tweet_text,
                            "created_at": time.ctime(),
                            "author_id": "bot",
                            "content_type": best_suggestion.content_type,
                            "source_urls": json.dumps(best_suggestion.source_urls),
                            "topics": json.dumps(best_suggestion.topics)
                        })
                        
                        # Store metrics
                        db["post_metrics"].insert({
                            "tweet_id": new_tweet_id,
                            "confidence": best_suggestion.confidence,
                            "personality_alignment": best_suggestion.personality_alignment,
                            "timing_score": best_suggestion.timing_score,
                            "expected_engagement": best_suggestion.expected_engagement
                        })
            
            # 15. Dynamic sleep based on state
            base_sleep = random.randint(30, 90)
            personality_factor = 1.0
            
            # Adjust timing based on personality and activity
            dominant_traits = personality.current_mode.get_dominant_traits()
            if MoodType.BLESSED in dominant_traits or MoodType.CURSED in dominant_traits:
                personality_factor = 0.7
            elif MoodType.INTELLECTUAL in dominant_traits:
                personality_factor = 1.3
                
            # Reduce sleep time if there's active engagement
            if reactive_system.active_conversations:
                personality_factor *= 0.8
                
            # Further reduce if we have high-confidence discoveries
            if any(d['confidence'] > 0.8 for d in discoveries):
                personality_factor *= 0.9
                
            # Adjust based on energy levels
            if personality.energy.mental_energy < 0.5:
                personality_factor *= 1.2
            if personality.energy.social_energy < 0.5:
                personality_factor *= 1.1
                
            sleep_time = int(base_sleep * personality_factor)
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in agent loop: {e}")
            
            # Store error in memory
            memory_system.add_episode(
                type="error",
                content={
                    'error': str(e),
                    'traceback': traceback.format_exc()
                },
                context={
                    'personality_state': personality.get_state_summary(),
                    'timeline_analysis': timeline_analysis
                },
                participants=set()
            )
            
            await asyncio.sleep(300)  # Sleep 5 minutes on error

@cli.command()
def run_memory():
    """Run memory system tests."""
    async def test_memory():
        db = init_db()
        rag = EnhancedRAG()
        memory_system = MemorySystem(db, rag.vector_store)
        
        # Test adding episodes
        memory_system.add_episode(
            type="test",
            content={
                'text': "This is a test episode",
                'topics': {'test': 1.0, 'memory': 0.8},
                'emotions': {'sentiment': 0.7, 'joy': 0.8}
            },
            context={'test': True},
            participants={'test_user'}
        )
        
        # Test relationship tracking
        relationship = memory_system.get_relationship('test_user')
        if relationship:
            click.echo("\nRelationship State:")
            click.echo(f"Trust Score: {relationship.trust_score:.2f}")
            click.echo(f"Influence Score: {relationship.influence_score:.2f}")
            click.echo(f"Topics: {json.dumps(relationship.topics, indent=2)}")
            
        # Test semantic memory
        concepts = memory_system.get_related_concepts('test')
        click.echo("\nRelated Concepts:")
        for concept, strength in concepts:
            click.echo(f"{concept}: {strength:.2f}")
            
        # Test memory consolidation
        memory_system.consolidate_memory()
        
        # Display working memory
        click.echo("\nWorking Memory:")
        for episode_id in memory_system.working_memory:
            episode = memory_system.episodes[episode_id]
            click.echo(f"\nEpisode {episode_id}:")
            click.echo(f"Type: {episode.type}")
            click.echo(f"Importance: {episode.importance:.2f}")
            click.echo(f"Attention Weight: {memory_system.attention_weights[episode_id]:.2f}")
            
    asyncio.run(test_memory())

###############################################################################
# 11. CLI Commands
###############################################################################

@click.group()
def cli():
    """Enhanced Twitter Bot CLI with RAG and Personality."""

@cli.command()
def setup_db():
    """Initialize the enhanced DB schema."""
    init_db()
    click.echo("Database initialized with enhanced schema.")

@cli.command()
def run_once():
    """Run a single iteration with all enhancements."""
    asyncio.run(single_iteration())

async def single_iteration():
    """Single iteration of the enhanced loop for testing."""
    db = init_db()
    rag = EnhancedRAG()
    memory_system = MemorySystem(db, rag.vector_store)
    personality = PersonalitySystem(db)
    vibe_analyzer = VibeAnalyzer()
    prompt_manager = PromptManager()
    twitter_api = get_twitter_api()
    twitter_actions = TwitterActions(twitter_api)
    content_generator = ContentGenerator(
        memory_system=memory_system,
        personality=personality,
        anthropic_client=get_anthropic_client()
    )
    researcher = ResearchManager(
        memory_system=memory_system,
        content_fetcher=ContentFetcher(
            markdowner_api_key=os.getenv("MARKDOWNER_API_KEY"),
            memory_system=memory_system,
            anthropic_client=get_anthropic_client()
        ),
        personality=personality,
        anthropic_client=get_anthropic_client(),
        search_config=SearchConfig(
            api_key=os.getenv("SCALESERP_API_KEY"),
            max_results=10
        )
    )
    
    # Get timeline data
    api = get_twitter_api()
    public_tweets = api.home_timeline(count=5)
    tweet_data = [{
        "id": str(t.id),
        "text": t.text,
        "author": t.user.screen_name,
        "created_at": str(t.created_at)
    } for t in public_tweets]
    
    # Analyze and display results
    timeline_analysis = vibe_analyzer.analyze_timeline(tweet_data, personality)
    click.echo("\nTimeline Analysis:")
    click.echo(json.dumps(timeline_analysis, indent=2))
    
    rag_context = rag.get_contextual_memory(
        "\n".join(t["text"] for t in tweet_data),
        personality,
        db
    )
    click.echo("\nRAG Context:")
    click.echo(json.dumps(rag_context, indent=2))
    
    # Test personality state
    personality.update_from_timeline(timeline_analysis)
    click.echo("\nPersonality State:")
    state = personality.get_state_summary()
    click.echo(f"Mode: {json.dumps(state['mode'], indent=2)}")
    click.echo(f"Energy: {json.dumps(state['energy'], indent=2)}")
    click.echo(f"Dominant Traits: {state['dominant_traits']}")
    
    # Test follow manager
    decision = await FollowManager(db, twitter_api).evaluate_account_for_follow("test_account")
    click.echo("\nFollow Decision:")
    click.echo(json.dumps(decision, indent=2))

@cli.command()
def run_forever():
    """Run the enhanced agentic loop indefinitely."""
    asyncio.run(enhanced_agentic_loop())

###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    cli()
