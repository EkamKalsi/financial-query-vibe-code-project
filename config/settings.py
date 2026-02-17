"""
Central configuration for the Reddit Financial Query multi-agent system.
Tunable values are loaded from environment variables (via .env file).
No API keys or OAuth credentials are required.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root is one level above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# HTTP client identity
# Reddit requires a descriptive User-Agent; requests without one are blocked.
# ---------------------------------------------------------------------------
REDDIT_USER_AGENT: str = os.getenv(
    "REDDIT_USER_AGENT", "FinancialQueryBot/0.1 (research project)"
)

# ---------------------------------------------------------------------------
# Scraper defaults
# ---------------------------------------------------------------------------
DEFAULT_SUBREDDITS: list[str] = [
    "investing",
    "stocks",
    "wallstreetbets",
    "personalfinance",
    "SecurityAnalysis",
    "ValueInvesting",
]

# Maximum posts to fetch per subreddit run (Reddit paginates at 100/page)
MAX_POSTS_PER_SUBREDDIT: int = int(os.getenv("MAX_POSTS_PER_SUBREDDIT", "500"))

# How many top-level comments to collect per post (0 = all)
MAX_COMMENTS_PER_POST: int = int(os.getenv("MAX_COMMENTS_PER_POST", "50"))

# Minimum score (upvotes) a post must have to be included
MIN_POST_SCORE: int = int(os.getenv("MIN_POST_SCORE", "10"))

# Sort order when fetching posts: hot | new | top | rising
DEFAULT_SORT: str = os.getenv("DEFAULT_SORT", "hot")

# Time filter used with sort="top": hour|day|week|month|year|all
DEFAULT_TIME_FILTER: str = os.getenv("DEFAULT_TIME_FILTER", "week")

# Seconds to sleep between requests — keeps us well under Reddit's ~1 req/s limit
REQUEST_DELAY_SECONDS: float = float(os.getenv("REQUEST_DELAY_SECONDS", "1.1"))

# Seconds to wait as the base for exponential back-off on 429 responses
RATE_LIMIT_BACKOFF_SECONDS: float = float(
    os.getenv("RATE_LIMIT_BACKOFF_SECONDS", "5.0")
)

# Max retry attempts on transient network / rate-limit errors
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Embedding agent
# ---------------------------------------------------------------------------

# sentence-transformers model used for embedding posts and queries.
# all-MiniLM-L6-v2: 80 MB, 384-dim, fast on CPU, good semantic quality.
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# Number of top comments to include when building the text to embed per post.
# More comments = richer signal but slower embedding and larger context later.
TOP_COMMENTS_TO_EMBED: int = int(os.getenv("TOP_COMMENTS_TO_EMBED", "5"))

# Batch size for sentence-transformers .encode() — tune down if RAM is tight
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# ---------------------------------------------------------------------------
# Query agent
# ---------------------------------------------------------------------------

# Anthropic API key — set this in .env
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Claude model used for answering queries
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# Number of top posts to retrieve and pass as context to the LLM
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# Max tokens for the LLM answer
QUERY_MAX_TOKENS: int = int(os.getenv("QUERY_MAX_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
ARCHIVE_DIR: Path = PROJECT_ROOT / "data" / "archive"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# Files written by the embedding agent and read by the query agent
EMBEDDINGS_FILE: Path = PROCESSED_DATA_DIR / "embeddings.npy"
POSTS_INDEX_FILE: Path = PROCESSED_DATA_DIR / "posts_index.json"

# Ensure directories exist
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARCHIVE_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
