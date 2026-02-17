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
> *"What risks do people associate with index fund investing in 2025?"*
> *"Summarise the sentiment around high-yield savings accounts this week."*

Rather than keyword search, the system uses **semantic retrieval** (embedding
similarity) combined with **LLM reasoning** to synthesise answers grounded in
actual posts and comments.

---

## Architecture

```
Reddit .json endpoint (no API key required)
    │
    ▼
┌─────────────────┐
│ Scraping Agent  │  Pulls posts + comments from investment subreddits via
│                 │  Reddit's public .json API. Paginates, rate-limits, and
│                 │  stores raw JSON to data/raw/.
└────────┬────────┘
         │ raw JSON (timestamped per run)
         ▼
┌──────────────────┐
│ Embedding Agent  │  Builds a text representation per post (title + body +
│                  │  top comments), generates 384-dim unit vectors using
│                  │  sentence-transformers (all-MiniLM-L6-v2), and persists
│                  │  embeddings.npy + posts_index.json to data/processed/.
│                  │  Supports incremental updates — only new posts are
│                  │  embedded on each run.
└────────┬─────────┘
         │ embeddings.npy  +  posts_index.json
         ▼
┌──────────────────┐
│  Query Agent     │  Embeds the user's question, computes cosine similarity
│                  │  against the index (dot product of unit vectors), and
│  (coming soon)   │  passes the top-k posts to Claude as context.
└────────┬─────────┘
         │ grounded LLM response + citations
         ▼
    User answer
```

The **Orchestrator** (`orchestrator.py`) ties the scraping and embedding agents
into a weekly cron pipeline:

```
[cron: every Monday 03:00]
        │
        ▼
  scrape top/week posts
        │
        ▼
  incremental embed (new posts only)
        │
        ▼
  archive raw JSON → data/archive/YYYY_WNN.zip
  (raw files removed from data/raw/ after zipping)
```

---

## Project structure

```
financial_query_vibe_code_project/
│
├── agents/
│   ├── scraping_agent.py     # Phase 1 — Reddit data collection         ✅
│   └── embedding_agent.py    # Phase 2 — vector embedding + index        ✅
│
├── config/
│   └── settings.py           # Centralised config, loaded from .env
│
├── data/
│   ├── raw/                  # JSON files from current scraping run
│   ├── processed/            # embeddings.npy + posts_index.json
│   └── archive/              # Weekly zip archives (YYYY_WNN.zip)
│
├── logs/                     # Runtime logs (one file per orchestrator run)
│
├── orchestrator.py           # Weekly pipeline: scrape → embed → archive  ✅
├── .env.example              # Tuning template (no credentials needed)
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

- Python 3.9+
- No Reddit account or API key needed

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure (optional)

```bash
cp .env.example .env
```

All values in `.env` are optional tuning knobs — the system works out of the
box with no credentials. The only value worth setting is a descriptive
`REDDIT_USER_AGENT` (Reddit may throttle requests with a generic one):

```
REDDIT_USER_AGENT=FinancialQueryBot/0.1 (your description here)
```

### 4. Run the scraper

```bash
# Quick test — 10 posts per subreddit, top of the week
python -m agents.scraping_agent investing stocks wallstreetbets \
    --sort top --time-filter week --limit 10

# Production run — all configured subreddits
python -m agents.scraping_agent investing stocks wallstreetbets \
    personalfinance SecurityAnalysis ValueInvesting \
    --sort top --time-filter week --limit 500 --min-score 10
```

Output JSON files are written to `data/raw/` with the naming pattern
`{subreddit}_{sort}_{UTC-timestamp}.json`.

### 5. Build the embedding index

```bash
# First run — full build from all raw files
python -m agents.embedding_agent

# Subsequent runs — only embeds posts not already in the index
# (called automatically by the orchestrator)
```

Output: `data/processed/embeddings.npy` (float32, shape `N × 384`) and
`data/processed/posts_index.json` (metadata aligned row-for-row).

### 6. Run the weekly pipeline (orchestrator)

```bash
python -m orchestrator
```

Or via cron (every Monday at 03:00):

```
0 3 * * 1  /path/to/.venv/bin/python -m orchestrator >> /path/to/logs/cron.log 2>&1
```

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

## Embedding details

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Dimensions | 384 |
| Normalisation | L2 unit vectors — dot product = cosine similarity |
| Storage | `data/processed/embeddings.npy` (float32) |
| Text per post | title + body + top-5 comments |
| Incremental | Yes — only new post IDs are embedded on each run |

---

## Rate limiting

Reddit's unauthenticated `.json` endpoint allows roughly one request per second.
The scraper enforces this with two layers:

1. **Fixed delay** — `REQUEST_DELAY_SECONDS` (default `1.1 s`) is slept after
   every request, keeping throughput comfortably under the limit.
2. **Exponential back-off** — on HTTP 429 or 5xx responses the scraper waits
   `RATE_LIMIT_BACKOFF_SECONDS × 2^attempt + jitter` seconds before retrying,
   up to `MAX_RETRIES` attempts.

Both values are configurable via `.env` without touching code.

---

## Roadmap

- [x] Phase 1 — Scraping Agent (Reddit `.json` endpoint, no API key)
- [x] Phase 2 — Embedding Agent (sentence-transformers, incremental updates)
- [x] Orchestrator (weekly cron pipeline with raw-file archiving)
- [ ] Phase 3 — Query Agent (cosine retrieval + Claude LLM response)
- [ ] Phase 4 — Simple query CLI / web UI

---

## Disclaimer

This project is for **research and educational purposes only**. Nothing
generated by this system constitutes financial advice. Always do your own
research before making investment decisions.
