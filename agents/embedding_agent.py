"""
Embedding Agent
===============
Reads all raw JSON files produced by the scraping agent, builds a single
text representation for each post, generates vector embeddings using
sentence-transformers, and persists two files to data/processed/:

  embeddings.npy    — float32 array of shape (N, embedding_dim)
  posts_index.json  — list of N metadata dicts (one per embedded post),
                      each containing enough info to display a citation
                      and reconstruct the text sent to the LLM.

Two entry points:
  build_index()   — full rebuild from all raw files (first run / manual reset)
  update_index()  — incremental: only embeds posts not already in the index,
                    then appends to existing files. Used by the orchestrator.

Run manually:
  python -m agents.embedding_agent
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    POSTS_INDEX_FILE,
    RAW_DATA_DIR,
    TOP_COMMENTS_TO_EMBED,
)

logger = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------------

def build_post_text(post: dict[str, Any], top_n_comments: int = TOP_COMMENTS_TO_EMBED) -> str:
    """
    Combine title, body, and top comments into a single string to embed.

    The model sees:
        <title>
        <body>          (omitted if empty / link post)
        Comments:
        - <comment 1>
        - <comment 2>
        ...

    Keeping comments in is important — a lot of the investment insight
    lives in the discussion, not the post body.
    """
    parts: list[str] = [post["title"]]

    body = (post.get("body") or "").strip()
    # Reddit stores "[removed]" or "" for deleted/link posts — skip those
    if body and body not in ("[removed]", "[deleted]"):
        parts.append(body)

    comments = post.get("comments", [])[:top_n_comments]
    if comments:
        comment_lines = [
            f"- {c['body'].strip()}"
            for c in comments
            if c.get("body") and c["body"] not in ("[removed]", "[deleted]")
        ]
        if comment_lines:
            parts.append("Comments:\n" + "\n".join(comment_lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_all_posts(raw_dir: Path = RAW_DATA_DIR) -> list[dict[str, Any]]:
    """
    Read every JSON file in *raw_dir* and return a deduplicated flat list
    of posts. Deduplication is by post `id` — if the same post appears in
    multiple scrape runs (e.g. it was top of the week twice) we keep the
    version with the higher score (most recent upvote count).
    """
    seen: dict[str, dict[str, Any]] = {}  # post_id -> post

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {raw_dir}. Run the scraping agent first."
        )

    logger.info("Loading raw data from %d file(s) in %s …", len(json_files), raw_dir)

    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s — could not parse: %s", path.name, exc)
            continue

        for post in data.get("posts", []):
            pid = post.get("id")
            if not pid:
                continue
            # Keep higher-score version on collision
            if pid not in seen or post.get("score", 0) > seen[pid].get("score", 0):
                seen[pid] = post

    posts = list(seen.values())
    logger.info("Loaded %d unique posts after deduplication.", len(posts))
    return posts


# ---------------------------------------------------------------------------
# Core embedding logic
# ---------------------------------------------------------------------------

def embed_posts(
    posts: list[dict[str, Any]],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    top_n_comments: int = TOP_COMMENTS_TO_EMBED,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Embed *posts* and return:
      - embeddings : float32 ndarray of shape (N, dim)
      - index      : list of metadata dicts aligned row-for-row with embeddings

    The metadata dict stored per post contains everything the query agent
    needs to display a citation and build the LLM context window.
    """
    logger.info("Loading embedding model '%s' …", model_name)
    model = SentenceTransformer(model_name)

    texts: list[str] = []
    index: list[dict[str, Any]] = []

    for post in posts:
        text = build_post_text(post, top_n_comments)
        if not text.strip():
            continue  # skip fully empty posts

        texts.append(text)
        index.append({
            "id": post["id"],
            "subreddit": post.get("subreddit", ""),
            "title": post.get("title", ""),
            "permalink": post.get("permalink", ""),
            "score": post.get("score", 0),
            "upvote_ratio": post.get("upvote_ratio", 0.0),
            "created_iso": post.get("created_iso", ""),
            "author": post.get("author", "[deleted]"),
            "flair": post.get("flair"),
            "embedded_text": text,  # stored so query agent can pass it to LLM
        })

    logger.info("Embedding %d posts in batches of %d …", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors → dot product == cosine sim
    )

    return embeddings.astype(np.float32), index


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(
    embeddings: np.ndarray,
    index: list[dict[str, Any]],
    embeddings_file: Path = EMBEDDINGS_FILE,
    posts_index_file: Path = POSTS_INDEX_FILE,
) -> None:
    """Persist embeddings array and metadata index to disk."""
    np.save(embeddings_file, embeddings)
    logger.info("Saved embeddings %s → %s", embeddings.shape, embeddings_file)

    posts_index_file.write_text(
        json.dumps(
            {
                "meta": {
                    "post_count": len(index),
                    "embedding_dim": embeddings.shape[1],
                    "model": EMBEDDING_MODEL,
                    "built_at": datetime.now(tz=timezone.utc).isoformat(),
                },
                "posts": index,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Saved post index (%d entries) → %s", len(index), posts_index_file)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_index(
    raw_dir: Path = RAW_DATA_DIR,
    embeddings_file: Path = EMBEDDINGS_FILE,
    posts_index_file: Path = POSTS_INDEX_FILE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Full pipeline: load raw posts → embed → save.
    Returns (embeddings, index) so callers can inspect without re-loading.
    """
    posts = load_all_posts(raw_dir)
    embeddings, index = embed_posts(posts)
    save(embeddings, index, embeddings_file, posts_index_file)
    return embeddings, index


def update_index(
    raw_dir: Path = RAW_DATA_DIR,
    embeddings_file: Path = EMBEDDINGS_FILE,
    posts_index_file: Path = POSTS_INDEX_FILE,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """
    Incremental update: embed only posts not already in the index, then
    append to the existing embeddings.npy and posts_index.json.

    Falls back to a full build_index() when no existing index is found
    (i.e. first run).

    Returns
    -------
    embeddings  : full merged float32 array of shape (N, dim)
    index       : full merged list of metadata dicts
    new_count   : number of newly embedded posts this run (0 if nothing new)
    """
    # ------------------------------------------------------------------ #
    # Load existing state (if any)
    # ------------------------------------------------------------------ #
    existing_embeddings: np.ndarray | None = None
    existing_index: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    if embeddings_file.exists() and posts_index_file.exists():
        try:
            existing_data = json.loads(posts_index_file.read_text(encoding="utf-8"))
            existing_index = existing_data.get("posts", [])
            seen_ids = {p["id"] for p in existing_index}
            existing_embeddings = np.load(embeddings_file)
            logger.info(
                "Loaded existing index: %d posts, embeddings shape %s.",
                len(existing_index), existing_embeddings.shape,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load existing index (%s). Falling back to full rebuild.", exc
            )
            return (*build_index(raw_dir, embeddings_file, posts_index_file), -1)
    else:
        logger.info("No existing index found — running full build.")
        embeddings, index = build_index(raw_dir, embeddings_file, posts_index_file)
        return embeddings, index, len(index)

    # ------------------------------------------------------------------ #
    # Find new posts
    # ------------------------------------------------------------------ #
    all_posts = load_all_posts(raw_dir)
    new_posts = [p for p in all_posts if p.get("id") not in seen_ids]

    if not new_posts:
        logger.info("No new posts to embed. Index is already up to date.")
        return existing_embeddings, existing_index, 0

    logger.info("%d new post(s) to embed (skipping %d already indexed).",
                len(new_posts), len(seen_ids))

    # ------------------------------------------------------------------ #
    # Embed new posts and merge
    # ------------------------------------------------------------------ #
    new_embeddings, new_index_entries = embed_posts(new_posts)

    merged_embeddings = np.vstack([existing_embeddings, new_embeddings]).astype(np.float32)
    merged_index = existing_index + new_index_entries

    save(merged_embeddings, merged_index, embeddings_file, posts_index_file)

    logger.info(
        "Index updated: +%d posts → total %d, embeddings shape %s.",
        len(new_posts), len(merged_index), merged_embeddings.shape,
    )
    return merged_embeddings, merged_index, len(new_posts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _configure_logging()
    embeddings, index = build_index()
    print(f"\nDone. {len(index)} posts embedded into shape {embeddings.shape}.")
    print(f"  Embeddings : {EMBEDDINGS_FILE}")
    print(f"  Post index : {POSTS_INDEX_FILE}")
