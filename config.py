"""Configuration settings for the bot"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

# Paths
BASE_DIR = Path(__file__).parent
VECTOR_DB_PATH = BASE_DIR / "data" / "vectors"
DB_PATH = BASE_DIR / "data" / "memory" / "bot.db"

# Ensure directories exist
VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Twitter API Configuration
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Validate required Twitter credentials
required_twitter_vars = [
    "TWITTER_API_KEY",
    "TWITTER_API_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_SECRET",
    "TWITTER_BEARER_TOKEN"
]

missing_vars = [var for var in required_twitter_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required Twitter credentials: {', '.join(missing_vars)}")

# Bot Configuration
MAX_DAILY_TWEETS = 50
MAX_DAILY_FOLLOWS = 50
MAX_DAILY_LIKES = 100
AUTO_FOLLOW_THRESHOLD = 0.8
HIGH_ENGAGEMENT_THRESHOLD = 0.7
