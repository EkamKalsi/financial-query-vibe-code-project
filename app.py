"""
Reddit Financial Query — Streamlit UI
======================================
Run with:
  streamlit run app.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from agents.query_agent import answer, reload_index
from agents.scraping_agent import scrape_multiple_subreddits
from agents.embedding_agent import update_index
from config.settings import (
    DEFAULT_SUBREDDITS,
    POSTS_INDEX_FILE,
    EMBEDDINGS_FILE,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Reddit Financial Query",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_index_meta() -> dict | None:
    """Read posts_index.json and return a summary dict, or None if missing."""
    if not POSTS_INDEX_FILE.exists():
        return None
    try:
        data = json.loads(POSTS_INDEX_FILE.read_text(encoding="utf-8"))
        posts = data.get("posts", [])
        meta = data.get("meta", {})
        subreddit_counts = Counter(p["subreddit"] for p in posts)
        return {
            "total_posts": len(posts),
            "subreddit_counts": dict(subreddit_counts),
            "built_at": meta.get("built_at", "unknown"),
            "embedding_dim": meta.get("embedding_dim", "—"),
            "model": meta.get("model", "—"),
        }
    except Exception:
        return None


def _fmt_built_at(iso: str) -> str:
    """Convert ISO timestamp to a human-friendly string."""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return iso


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📈 Reddit Financial Query")
st.caption(
    "Ask questions about investing, markets, and personal finance — "
    "answered using real Reddit community discussions."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 1: Index Stats + Update
# ---------------------------------------------------------------------------

st.subheader("🗄️ Index Status")

meta = _load_index_meta()

if meta is None:
    st.warning(
        "No index found. Click **Fetch & Update Index** below to scrape posts "
        "and build the embedding index for the first time."
    )
else:
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", meta["total_posts"])
    col2.metric("Subreddits", len(meta["subreddit_counts"]))
    col3.metric("Embedding Dims", meta["embedding_dim"])
    col4.metric("Last Updated", _fmt_built_at(meta["built_at"]))

    # Per-subreddit breakdown
    st.markdown("**Posts per subreddit:**")
    sub_cols = st.columns(len(meta["subreddit_counts"]))
    for col, (subreddit, count) in zip(sub_cols, sorted(meta["subreddit_counts"].items())):
        col.metric(f"r/{subreddit}", count)

# Update button
st.markdown("")
if st.button("🔄 Fetch & Update Index", type="secondary"):
    progress = st.empty()
    progress.info("Scraping top posts from all subreddits…")
    try:
        paths = scrape_multiple_subreddits(
            DEFAULT_SUBREDDITS,
            sort="top",
            time_filter="week",
        )
        progress.info(f"Scraped {len(paths)} subreddit(s). Building embeddings…")
        _, _, new_count = update_index()
        reload_index()  # invalidate module-level cache
        progress.empty()
        st.success(
            f"Index updated — **{new_count}** new post(s) embedded. "
            f"Reload the page to see updated stats."
        )
    except Exception as exc:
        progress.empty()
        st.error(f"Update failed: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Query
# ---------------------------------------------------------------------------

st.subheader("💬 Ask a Question")

query = st.text_input(
    label="Your question",
    placeholder='e.g. "What are people saying about NVDA long-term?"',
    label_visibility="collapsed",
)

ask_col, _ = st.columns([1, 5])
ask_clicked = ask_col.button("Ask", type="primary", disabled=not query.strip())

# ---------------------------------------------------------------------------
# Section 3: Response
# ---------------------------------------------------------------------------

if ask_clicked and query.strip():
    if not EMBEDDINGS_FILE.exists():
        st.error("No index found. Please fetch and update the index first.")
    else:
        with st.spinner("Retrieving relevant posts and generating answer…"):
            try:
                result = answer(query.strip())
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()

        st.subheader("🤖 Answer")
        st.markdown(result.text)

        # Sources expander
        with st.expander(f"📎 Sources ({len(result.sources)} posts retrieved)", expanded=False):
            for s in result.sources:
                st.markdown(
                    f"**[{s['rank']}]** `sim={s['similarity']:.3f}` &nbsp;|&nbsp; "
                    f"[{s['title']}]({s['permalink']}) &nbsp;—&nbsp; "
                    f"r/{s['subreddit']} &nbsp;·&nbsp; score: {s['score']}"
                )