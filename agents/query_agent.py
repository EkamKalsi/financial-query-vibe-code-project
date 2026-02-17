"""
Query Agent
===========
Takes a natural-language financial question and returns a structured,
cited answer grounded in the indexed Reddit posts.

Pipeline
--------
  1. Load embeddings.npy + posts_index.json (once, on first call).
  2. Embed the user's query with the same sentence-transformers model.
  3. Rank all indexed posts by cosine similarity (dot product of unit vectors).
  4. Pass the top-k posts to Claude with a structured prompt.
  5. Return a QueryResponse: the LLM answer (markdown) + source metadata.

Usage
-----
  # Programmatic
  from agents.query_agent import answer
  result = answer("What are people saying about US debt levels?")
  print(result.text)

  # CLI
  python -m agents.query_agent "What are people saying about US debt levels?"
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import anthropic
from sentence_transformers import SentenceTransformer

from config.settings import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    POSTS_INDEX_FILE,
    QUERY_MAX_TOKENS,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# Index loading (lazy, cached at module level)
# ---------------------------------------------------------------------------

_embeddings: np.ndarray | None = None
_index: list[dict[str, Any]] | None = None
_embed_model: SentenceTransformer | None = None


def _load_index(
    embeddings_file: Path = EMBEDDINGS_FILE,
    posts_index_file: Path = POSTS_INDEX_FILE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load embeddings and post metadata from disk (cached after first call)."""
    global _embeddings, _index

    if _embeddings is not None and _index is not None:
        return _embeddings, _index

    if not embeddings_file.exists() or not posts_index_file.exists():
        raise FileNotFoundError(
            "Embedding index not found. Run the embedding agent first:\n"
            "  python -m agents.embedding_agent"
        )

    _embeddings = np.load(embeddings_file)
    data = json.loads(posts_index_file.read_text(encoding="utf-8"))
    _index = data["posts"]

    logger.info(
        "Loaded index: %d posts, embeddings shape %s.", len(_index), _embeddings.shape
    )
    return _embeddings, _index


def _get_embed_model() -> SentenceTransformer:
    """Return the shared embedding model (cached after first call)."""
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievedPost:
    post: dict[str, Any]
    similarity: float


def retrieve(
    query: str,
    top_k: int = TOP_K_RESULTS,
    embeddings_file: Path = EMBEDDINGS_FILE,
    posts_index_file: Path = POSTS_INDEX_FILE,
) -> list[RetrievedPost]:
    """
    Embed *query* and return the *top_k* most similar posts by cosine similarity.
    (Unit vectors → dot product == cosine similarity, no division needed.)
    """
    embeddings, index = _load_index(embeddings_file, posts_index_file)
    model = _get_embed_model()

    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    scores: np.ndarray = embeddings @ query_vec  # shape: (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        RetrievedPost(post=index[i], similarity=float(scores[i]))
        for i in top_indices
    ]


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_context(results: list[RetrievedPost]) -> str:
    """
    Build the context block passed to Claude.
    Each post gets a numbered header with metadata so the LLM can cite it.
    """
    blocks: list[str] = []
    for n, r in enumerate(results, start=1):
        p = r.post
        header = (
            f"[Post {n}] r/{p['subreddit']} | "
            f"Score: {p['score']} | "
            f"Similarity: {r.similarity:.3f}\n"
            f"Title: {p['title']}\n"
            f"Link: {p['permalink']}"
        )
        blocks.append(f"{header}\n\n{p.get('embedded_text', '').strip()}")

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a financial research assistant. You answer questions about investing,
markets, and personal finance by analysing discussions from Reddit investment
communities.

You will be given a user question and a set of relevant Reddit posts retrieved
by semantic search. Base your answer strictly on the provided posts — do not
add outside knowledge or make up facts.

Structure your response as follows:

## Summary
2–3 sentence overview of what the community is saying.

## Key Themes
Bullet-point list of the main arguments, views, or strategies mentioned.

## Sentiment
One of: Bullish / Bearish / Mixed / Neutral — with a one-sentence explanation.

## Notable Community Quotes
1–3 short, representative quotes from the posts (paraphrase or quote directly).

## Caveats & Risks
Any warnings, counterarguments, or risks the community highlighted.

## Sources
List each post used as: - [Title](permalink) — r/subreddit (score: N)

Keep your answer concise and grounded. If the retrieved posts do not contain
enough information to answer the question, say so clearly.\
"""


@dataclass
class QueryResponse:
    query: str
    text: str                          # full LLM response (markdown)
    sources: list[dict[str, Any]] = field(default_factory=list)  # retrieved post metadata
    model: str = ANTHROPIC_MODEL
    answered_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())


def answer(
    query: str,
    top_k: int = TOP_K_RESULTS,
    model: str = ANTHROPIC_MODEL,
    max_tokens: int = QUERY_MAX_TOKENS,
) -> QueryResponse:
    """
    Full pipeline: retrieve top-k posts → build context → call Claude → return response.

    Parameters
    ----------
    query     : The user's natural-language question.
    top_k     : Number of posts to retrieve and pass as context.
    model     : Anthropic model ID to use.
    max_tokens: Maximum tokens in the LLM response.

    Returns
    -------
    QueryResponse with the answer text and source metadata.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )

    # Step 1: Retrieve
    logger.info("Retrieving top-%d posts for query: %r", top_k, query)
    results = retrieve(query, top_k=top_k)
    logger.info(
        "Top result: %r (sim=%.3f)",
        results[0].post["title"][:60] if results else "—",
        results[0].similarity if results else 0.0,
    )

    # Step 2: Format context
    context = _format_context(results)

    user_message = f"User question: {query}\n\nRetrieved posts:\n\n{context}"

    # Step 3: Call Claude
    logger.info("Calling %s …", model)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text: str = message.content[0].text

    # Step 4: Pack sources
    sources = [
        {
            "rank": n,
            "subreddit": r.post["subreddit"],
            "title": r.post["title"],
            "permalink": r.post["permalink"],
            "score": r.post["score"],
            "similarity": round(r.similarity, 4),
        }
        for n, r in enumerate(results, start=1)
    ]

    return QueryResponse(query=query, text=response_text, sources=sources, model=model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _configure_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m agents.query_agent \"your question here\"")
        sys.exit(1)

    query_text = " ".join(sys.argv[1:])

    try:
        result = answer(query_text)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Query : {result.query}")
    print(f"Model : {result.model}  |  Answered at: {result.answered_at}")
    print(f"{'='*70}\n")
    print(result.text)
    print(f"\n{'─'*70}")
    print("Retrieved sources:")
    for s in result.sources:
        print(f"  [{s['rank']}] sim={s['similarity']:.3f} | r/{s['subreddit']} "
              f"(score={s['score']}) — {s['title'][:60]}")
