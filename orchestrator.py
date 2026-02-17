"""
Orchestrator
============
Weekly pipeline: scrape → incremental embed → archive raw files.

Designed to be invoked by cron once per week. Each run:
  1. Scrapes hot/top-of-the-week posts from all configured subreddits.
  2. Embeds only the new posts (not already in the index) and appends them.
  3. Zips this week's raw JSON files into data/archive/<YYYY_WNN>.zip
     and removes them from data/raw/, keeping that folder clean between runs.

Cron example (every Monday at 03:00 local time):
  0 3 * * 1  /path/to/.venv/bin/python -m orchestrator >> /path/to/logs/cron.log 2>&1

First-run note:
  If no existing index is found, the embedding step performs a full build
  instead of an incremental update. No special handling needed.
"""

from __future__ import annotations

import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.embedding_agent import update_index
from agents.scraping_agent import scrape_multiple_subreddits
from config.settings import (
    ARCHIVE_DIR,
    DEFAULT_SUBREDDITS,
    LOG_DIR,
    RAW_DATA_DIR,
)

# ---------------------------------------------------------------------------
# Logging — writes to both stdout and a dated log file
# ---------------------------------------------------------------------------

def _configure_logging(run_dt: datetime) -> None:
    log_file = LOG_DIR / f"orchestrator_{run_dt.strftime('%Y%m%dT%H%M%SZ')}.log"
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Archive helper
# ---------------------------------------------------------------------------

def _archive_raw_files(raw_dir: Path, archive_dir: Path, run_dt: datetime) -> Path | None:
    """
    Zip all JSON files currently in *raw_dir* into a weekly archive file
    inside *archive_dir*, then delete the originals.

    Archive name uses ISO week: e.g. data/archive/2026_W07.zip
    If a zip for the current week already exists (e.g. a re-run within the
    same week), new files are appended to the existing archive.

    Returns the path to the zip file, or None if there was nothing to archive.
    """
    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        logger.info("No raw JSON files to archive.")
        return None

    # %G = ISO year (handles Jan week-of-previous-year edge case)
    # %V = ISO week number, zero-padded
    iso_week = run_dt.strftime("%G_W%V")
    zip_path = archive_dir / f"{iso_week}.zip"

    logger.info(
        "Archiving %d file(s) → %s …", len(json_files), zip_path.name
    )

    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in json_files:
            # arcname stores only the filename, not the full path
            zf.write(f, arcname=f.name)
            logger.debug("  Zipped %s", f.name)

    # Remove originals only after the zip is successfully closed
    for f in json_files:
        f.unlink()

    logger.info("Archived and removed %d raw file(s). Zip: %s", len(json_files), zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(subreddits: list[str] | None = None) -> dict[str, Any]:
    """
    Execute the full weekly pipeline and return a summary dict.

    Parameters
    ----------
    subreddits : list of subreddit names to scrape. Defaults to
                 DEFAULT_SUBREDDITS from config/settings.py.
    """
    run_dt = datetime.now(tz=timezone.utc)
    _configure_logging(run_dt)

    if subreddits is None:
        subreddits = DEFAULT_SUBREDDITS

    logger.info("=== Orchestrator run started — %s ===", run_dt.isoformat())
    logger.info("Subreddits: %s", subreddits)

    summary: dict[str, Any] = {
        "run_at": run_dt.isoformat(),
        "subreddits": subreddits,
        "scraped_files": [],
        "new_posts_embedded": 0,
        "total_index_size": 0,
        "archive_path": None,
        "errors": [],
    }

    # ------------------------------------------------------------------
    # Step 1: Scrape
    # ------------------------------------------------------------------
    logger.info("--- Step 1: Scraping ---")
    try:
        paths = scrape_multiple_subreddits(
            subreddits,
            sort="top",
            time_filter="week",
        )
        summary["scraped_files"] = [str(p) for p in paths]
        logger.info("Scraped %d subreddit(s) → %d file(s).", len(subreddits), len(paths))
    except Exception as exc:  # noqa: BLE001
        msg = f"Scraping failed: {exc}"
        logger.error(msg)
        summary["errors"].append(msg)
        # Abort early — no point embedding if we have no new data
        return summary

    # ------------------------------------------------------------------
    # Step 2: Incremental embedding
    # ------------------------------------------------------------------
    logger.info("--- Step 2: Incremental embedding ---")
    try:
        _embeddings, index, new_count = update_index()
        summary["new_posts_embedded"] = new_count
        summary["total_index_size"] = len(index)
        logger.info(
            "Embedding complete. New: %d  |  Total index: %d",
            new_count, len(index),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Embedding failed: {exc}"
        logger.error(msg)
        summary["errors"].append(msg)
        # Raw files are still present — don't archive so they can be
        # retried on the next run.
        return summary

    # ------------------------------------------------------------------
    # Step 3: Archive raw files
    # ------------------------------------------------------------------
    logger.info("--- Step 3: Archiving raw files ---")
    try:
        zip_path = _archive_raw_files(RAW_DATA_DIR, ARCHIVE_DIR, run_dt)
        summary["archive_path"] = str(zip_path) if zip_path else None
    except Exception as exc:  # noqa: BLE001
        # Non-fatal — index is already updated; log and continue.
        msg = f"Archiving failed (raw files left in place): {exc}"
        logger.error(msg)
        summary["errors"].append(msg)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=== Run complete ===")
    logger.info("  New posts embedded : %d", summary["new_posts_embedded"])
    logger.info("  Total index size   : %d", summary["total_index_size"])
    logger.info("  Archive            : %s", summary["archive_path"])
    if summary["errors"]:
        logger.warning("  Errors             : %s", summary["errors"])

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run()
    if result["errors"]:
        sys.exit(1)
