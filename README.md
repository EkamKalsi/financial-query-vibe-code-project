# Reddit Financial Query — Multi-Agent RAG System

A multi-agent pipeline that scrapes investment discussions from Reddit,
indexes them with vector embeddings, and answers natural-language financial
queries by retrieving and reasoning over real community insight.

---

## What this project does

Retail investors share a large volume of analysis, risk assessment, and
strategy discussion across Reddit. This system mines that content and makes
it queryable — you can ask questions like:

> *"What are the most common arguments for and against holding $NVDA long-term?"*
> *"What risks do people associate with index fund investing in 2024?"*
> *"Summarise the sentiment around high-yield savings accounts this week."*

Rather than keyword search, the system uses **semantic retrieval** (embedding
similarity) combined with **LLM reasoning** to synthesise answers grounded in
actual posts and comments.

---

## Architecture — five agents

```
Reddit API
    │
    ▼
┌─────────────────┐
│  Scraping Agent │  Pulls posts + comments from investment subreddits via PRAW.
│                 │  Handles rate limits, stores raw JSON to data/raw/.
└────────┬────────┘
         │ raw JSON
         ▼
┌─────────────────┐
│ Indexing Agent  │  Cleans text, generates vector embeddings
│                 │  (sentence-transformers), builds a FAISS index.
└────────┬────────┘
         │ vector index + metadata
         ▼
┌──────────────────┐
│ Retrieval Agent  │  Takes a user query, embeds it, performs ANN search,
│                  │  returns the top-k most relevant posts/comments.
└────────┬─────────┘
         │ retrieved context
         ▼
┌──────────────────┐
│ Reasoning Agent  │  Analyses retrieved content for patterns, risks,
│                  │  and sentiment using an LLM (Claude / GPT-4).
└────────┬─────────┘
         │ structured analysis
         ▼
┌──────────────────┐
│ Answering Agent  │  Synthesises a concise, user-friendly response
│                  │  with citations back to source posts.
└──────────────────┘
         │
         ▼
    User answer
```

---

## Project structure

```
financial_query_vibe_code_project/
│
├── agents/
│   ├── scraping_agent.py     # Phase 1 — Reddit data collection  ✅
│   ├── indexing_agent.py     # Phase 2 — embedding + FAISS index  🔜
│   ├── retrieval_agent.py    # Phase 3 — semantic search           🔜
│   ├── reasoning_agent.py    # Phase 4 — LLM pattern analysis      🔜
│   └── answering_agent.py    # Phase 5 — response generation       🔜
│
├── config/
│   └── settings.py           # Centralised config, loaded from .env
│
├── data/
│   ├── raw/                  # JSON files from scraping agent
│   └── processed/            # Cleaned text + FAISS indexes
│
├── logs/                     # Runtime logs
│
├── .env.example              # Credential + tuning template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Subreddits targeted

| Subreddit | Focus |
|---|---|
| r/investing | Long-term strategies, fundamentals |
| r/stocks | Individual stock analysis and news |
| r/wallstreetbets | High-risk trades, options, sentiment |
| r/personalfinance | Budgeting, savings, retirement |
| r/SecurityAnalysis | Deep fundamental / value analysis |
| r/ValueInvesting | Buffett-style long-horizon investing |

---

## Quick start

### 1. Prerequisites

- Python 3.10+
- A Reddit account with a **script**-type app created at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
REDDIT_CLIENT_ID=<the short string below your app name>
REDDIT_CLIENT_SECRET=<the string next to "secret">
REDDIT_USER_AGENT=FinancialQueryBot/0.1 by YourRedditUsername
```

### 4. Run the scraper

```bash
# Fetch top 100 posts this week from two subreddits
python -m agents.scraping_agent investing wallstreetbets \
    --sort top --time-filter week --limit 100

# Fetch latest 200 posts from a single subreddit
python -m agents.scraping_agent stocks --sort new --limit 200
```

Output JSON files are written to `data/raw/`.

---

## Scraper CLI reference

| Flag | Default | Description |
|---|---|---|
| `--sort` | `hot` | `hot` / `new` / `top` / `rising` |
| `--time-filter` | `week` | Used with `--sort top`: `hour/day/week/month/year/all` |
| `--limit` | `500` | Max posts per subreddit (Reddit cap: 1000) |
| `--min-score` | `10` | Skip posts below this upvote count |
| `--max-comments` | `50` | Top-level comments collected per post |

---

## Rate limiting

PRAW automatically respects Reddit's OAuth rate-limit headers (60 requests /
minute for script apps). The scraper adds an additional exponential back-off
layer with jitter for HTTP 429 responses and transient network errors, up to
`MAX_RETRIES` attempts.

---

## Roadmap

- [x] Phase 1 — Scraping Agent
- [ ] Phase 2 — Indexing Agent (sentence-transformers + FAISS)
- [ ] Phase 3 — Retrieval Agent (ANN semantic search)
- [ ] Phase 4 — Reasoning Agent (LLM analysis)
- [ ] Phase 5 — Answering Agent (response synthesis + citations)
- [ ] Phase 6 — Simple query CLI / web UI

---

## Disclaimer

This project is for **research and educational purposes only**. Nothing
generated by this system constitutes financial advice. Always do your own
research before making investment decisions.
