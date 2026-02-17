"""
Scraping Agent
==============
Fetches posts and comments from Reddit investment subreddits using Reddit's
public .json endpoint — no OAuth, no API keys required.

Endpoints used
--------------
  Listings : https://www.reddit.com/r/{sub}/{sort}.json?limit=100&after=...
  Comments : https://www.reddit.com/r/{sub}/comments/{id}.json?limit=N&depth=1

Rate-limit strategy
-------------------
Reddit's unauthenticated endpoint allows roughly 1 request/second.
This module enforces a configurable inter-request delay and adds
exponential back-off with jitter on HTTP 429 / 5xx responses.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from config.settings import (
    DEFAULT_SORT,
    DEFAULT_TIME_FILTER,
    MAX_COMMENTS_PER_POST,
    MAX_POSTS_PER_SUBREDDIT,
    MAX_RETRIES,
    MIN_POST_SCORE,
    RATE_LIMIT_BACKOFF_SECONDS,
    RAW_DATA_DIR,
    REQUEST_DELAY_SECONDS,
    REDDIT_USER_AGENT,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# HTTP client factory
# ---------------------------------------------------------------------------

def build_client() -> httpx.Client:
    """Return a configured httpx client with the Reddit-required User-Agent."""
    return httpx.Client(
        headers={"User-Agent": REDDIT_USER_AGENT},
        follow_redirects=True,
        timeout=15.0,
    )


# ---------------------------------------------------------------------------
# Resilient GET
# ---------------------------------------------------------------------------

def _get(client: httpx.Client, url: str, params: dict | None = None) -> Any:
    """
    Perform a GET request, retrying on 429 / 5xx with exponential back-off.
    Returns the parsed JSON body.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.get(url, params=params)

            if response.status_code == 429:
                wait = RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt) + random.random()
                logger.warning("Rate-limited (429). Waiting %.1f s…", wait)
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                if attempt == MAX_RETRIES:
                    response.raise_for_status()
                wait = (2 ** attempt) + random.random()
                logger.warning(
                    "Server error %d. Retry %d/%d in %.1f s…",
                    response.status_code, attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except httpx.RequestError as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = (2 ** attempt) + random.random()
            logger.warning(
                "Network error (%s). Retry %d/%d in %.1f s…",
                exc, attempt + 1, MAX_RETRIES, wait,
            )
            time.sleep(wait)

    raise RuntimeError("Exhausted retries.")


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_comment(raw: dict) -> dict[str, Any]:
    """Flatten a single comment's raw JSON data into a plain dict."""
    d = raw.get("data", {})
    created = d.get("created_utc", 0)
    return {
        "id": d.get("id"),
        "author": d.get("author", "[deleted]"),
        "body": d.get("body", ""),
        "score": d.get("score", 0),
        "created_utc": created,
        "created_iso": datetime.fromtimestamp(created, tz=timezone.utc).isoformat(),
        "is_submitter": d.get("is_submitter", False),
        "depth": d.get("depth", 0),
    }


def _extract_post(raw: dict, comments: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten a post's raw JSON data + pre-fetched comments into a plain dict."""
    d = raw.get("data", {})
    created = d.get("created_utc", 0)
    permalink = d.get("permalink", "")
    return {
        "id": d.get("id"),
        "subreddit": d.get("subreddit"),
        "title": d.get("title", ""),
        "body": d.get("selftext", ""),
        "url": d.get("url", ""),
        "permalink": f"https://www.reddit.com{permalink}",
        "author": d.get("author", "[deleted]"),
        "score": d.get("score", 0),
        "upvote_ratio": d.get("upvote_ratio", 0.0),
        "num_comments": d.get("num_comments", 0),
        "created_utc": created,
        "created_iso": datetime.fromtimestamp(created, tz=timezone.utc).isoformat(),
        "flair": d.get("link_flair_text"),
        "is_self": d.get("is_self", False),
        "comments": comments,
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Comment fetching
# ---------------------------------------------------------------------------

def _fetch_comments(
    client: httpx.Client,
    subreddit: str,
    post_id: str,
    max_comments: int,
) -> list[dict[str, Any]]:
    """
    Hit the post's .json endpoint and extract top-level comments.
    depth=1 means Reddit only returns direct replies to the post itself.
    """
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    params: dict[str, Any] = {"depth": 1, "showmore": 0}
    if max_comments:
        params["limit"] = max_comments

    try:
        data = _get(client, url, params=params)
        time.sleep(REQUEST_DELAY_SECONDS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch comments for post %s: %s", post_id, exc)
        return []

    # Reddit returns a 2-element list: [post_listing, comments_listing]
    if not isinstance(data, list) or len(data) < 2:
        return []

    children = data[1].get("data", {}).get("children", [])
    comments: list[dict[str, Any]] = []
    for child in children:
        if child.get("kind") != "t1":  # t1 = comment
            continue
        comments.append(_extract_comment(child))
        if max_comments and len(comments) >= max_comments:
            break

    return comments


# ---------------------------------------------------------------------------
# Listing pagination
# ---------------------------------------------------------------------------

def _iter_listing(
    client: httpx.Client,
    subreddit: str,
    sort: str,
    time_filter: str,
    limit: int,
) -> list[dict]:
    """
    Page through the subreddit listing until *limit* posts are collected
    or Reddit returns no more results.

    Reddit returns max 100 posts per request; we paginate using `after`.
    """
    base_url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    collected: list[dict] = []
    after: str | None = None
    page_size = min(100, limit)

    while len(collected) < limit:
        params: dict[str, Any] = {"limit": page_size}
        if sort == "top":
            params["t"] = time_filter
        if after:
            params["after"] = after

        data = _get(client, base_url, params=params)
        time.sleep(REQUEST_DELAY_SECONDS)

        listing = data.get("data", {})
        children = listing.get("children", [])

        if not children:
            break  # no more posts

        for child in children:
            if child.get("kind") == "t3":  # t3 = link/post
                collected.append(child)
            if len(collected) >= limit:
                break

        after = listing.get("after")
        if not after:
            break  # last page

    return collected


# ---------------------------------------------------------------------------
# Core scraping logic
# ---------------------------------------------------------------------------

def scrape_subreddit(
    client: httpx.Client,
    subreddit_name: str,
    *,
    sort: str = DEFAULT_SORT,
    time_filter: str = DEFAULT_TIME_FILTER,
    limit: int = MAX_POSTS_PER_SUBREDDIT,
    min_score: int = MIN_POST_SCORE,
    max_comments: int = MAX_COMMENTS_PER_POST,
    output_dir: Path = RAW_DATA_DIR,
) -> Path:
    """
    Fetch posts from *subreddit_name* and persist them as a JSON file.

    Parameters
    ----------
    client          : Shared httpx.Client instance.
    subreddit_name  : e.g. "investing"
    sort            : "hot" | "new" | "top" | "rising"
    time_filter     : Used only when sort="top". "hour|day|week|month|year|all"
    limit           : Max posts to retrieve (Reddit paginates at 100/page).
    min_score       : Skip posts below this upvote count.
    max_comments    : Top-level comments to collect per post (0 = all).
    output_dir      : Directory where JSON files are written.

    Returns
    -------
    Path to the written JSON file.
    """
    logger.info(
        "Scraping r/%s  sort=%s  limit=%d  min_score=%d",
        subreddit_name, sort, limit, min_score,
    )

    raw_posts = _iter_listing(client, subreddit_name, sort, time_filter, limit)

    posts: list[dict[str, Any]] = []
    skipped = 0

    for raw in raw_posts:
        post_data_raw = raw.get("data", {})
        score = post_data_raw.get("score", 0)

        if score < min_score:
            skipped += 1
            continue

        post_id = post_data_raw.get("id", "")
        title = post_data_raw.get("title", "")[:60]
        logger.debug("  → [%s] %s (score=%d)", post_id, title, score)

        comments = _fetch_comments(client, subreddit_name, post_id, max_comments)
        posts.append(_extract_post(raw, comments))

    logger.info(
        "r/%s: fetched %d posts, skipped %d below min_score=%d.",
        subreddit_name, len(posts), skipped, min_score,
    )

    # ----- Persist ----------------------------------------------------------
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{subreddit_name}_{sort}_{timestamp}.json"
    output_path = output_dir / filename

    output_path.write_text(
        json.dumps(
            {
                "meta": {
                    "subreddit": subreddit_name,
                    "sort": sort,
                    "time_filter": time_filter if sort == "top" else None,
                    "limit": limit,
                    "min_score": min_score,
                    "max_comments": max_comments,
                    "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
                    "post_count": len(posts),
                },
                "posts": posts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info("Saved %d posts → %s", len(posts), output_path)
    return output_path


def scrape_multiple_subreddits(
    subreddits: list[str],
    **kwargs,
) -> list[Path]:
    """
    Convenience wrapper: scrape several subreddits sequentially,
    sharing a single httpx.Client across all requests.
    """
    results: list[Path] = []
    with build_client() as client:
        for name in subreddits:
            try:
                path = scrape_subreddit(client, name, **kwargs)
                results.append(path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to scrape r/%s: %s", name, exc)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Reddit investment subreddit scraper (no API key required)"
    )
    parser.add_argument(
        "subreddits",
        nargs="+",
        help="One or more subreddit names (without r/ prefix)",
    )
    parser.add_argument(
        "--sort",
        default=DEFAULT_SORT,
        choices=["hot", "new", "top", "rising"],
        help="Listing sort order (default: %(default)s)",
    )
    parser.add_argument(
        "--time-filter",
        default=DEFAULT_TIME_FILTER,
        choices=["hour", "day", "week", "month", "year", "all"],
        help="Time filter for --sort=top (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_POSTS_PER_SUBREDDIT,
        help="Max posts per subreddit (default: %(default)s)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=MIN_POST_SCORE,
        help="Minimum post score to include (default: %(default)s)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=MAX_COMMENTS_PER_POST,
        help="Max top-level comments per post (default: %(default)s)",
    )

    args = parser.parse_args()

    paths = scrape_multiple_subreddits(
        args.subreddits,
        sort=args.sort,
        time_filter=args.time_filter,
        limit=args.limit,
        min_score=args.min_score,
        max_comments=args.max_comments,
    )

    print("\nOutput files:")
    for p in paths:
        print(f"  {p}")
