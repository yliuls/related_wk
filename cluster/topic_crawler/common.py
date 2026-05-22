from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import requests

from .schemas import normalize_record


LOGGER = logging.getLogger(__name__)
USER_AGENT = "topic-crawler/0.1 (+https://github.com/openai/codex)"


def ensure_dir(path: str | os.PathLike) -> None:
    """Create directory if needed."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def read_json_or_jsonl(path: str | os.PathLike) -> list[dict]:
    """Read JSON or JSONL. Support list[dict] and {'papers': [...]} formats."""
    path = str(path)
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("papers", "records", "data"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError(f"Unsupported JSON format: {path}")


def write_jsonl_atomic(records: list[dict], path: str | os.PathLike) -> None:
    """Write JSONL atomically via temp file + rename."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(normalize_record(record), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json_atomic(obj, path: str | os.PathLike) -> None:
    """Write JSON atomically."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(obj, list) and all(isinstance(item, dict) for item in obj):
        obj = [normalize_record(item) for item in obj]
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def fetch_text(
    url: str,
    cache_dir: str | None = None,
    force: bool = False,
    timeout: int = 30,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> str | None:
    """Fetch text with retry and optional URL-sha1 cache."""
    if not url:
        return None
    cache_path = None
    if cache_dir:
        ensure_dir(cache_dir)
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        cache_path = Path(cache_dir) / f"{key}.html"
        if cache_path.exists() and not force:
            return cache_path.read_text(encoding="utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if response.status_code in (403, 404):
                LOGGER.warning("Fetch skipped for %s: HTTP %s", url, response.status_code)
                return None
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            text = response.text
            if cache_path:
                cache_path.write_text(text, encoding="utf-8")
            return text
        except requests.RequestException as exc:
            LOGGER.warning("Fetch attempt %s/%s failed for %s: %s", attempt, max_retries, url, exc)
            if attempt < max_retries:
                time.sleep(sleep_seconds * attempt)
    return None


def normalize_title(title: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    text = (title or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def normalize_topic_text(topic: str | None) -> str | None:
    """Normalize topic text and arrows."""
    if topic is None:
        return None
    text = str(topic).strip()
    if not text:
        return None
    text = re.sub(r"\s*(?:->|→|>|/)\s*", " -> ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_hierarchical_topic(topic: str | None) -> tuple[str | None, str | None]:
    """Split A->B / A > B / A: B into primary and secondary."""
    text = normalize_topic_text(topic)
    if not text:
        return None, None
    for sep in (" -> ", ":"):
        if sep in text:
            parts = [part.strip() for part in text.split(sep, 1)]
            return parts[0] or None, parts[1] or None
    return text, None


def normalize_presentation_type(text: str | None) -> str:
    """Return oral / spotlight / poster / unknown."""
    value = str(text or "").strip().lower()
    if not value:
        return "unknown"
    if re.search(r"\boral\b|talk|plenary", value):
        return "oral"
    if "spotlight" in value or "highlight" in value:
        return "spotlight"
    if "poster" in value:
        return "poster"
    return "unknown"


def normalize_nlp_paper_type(text: str | None) -> str:
    """Return best_paper / main_paper / findings / workshop / demo / industry / unknown."""
    value = str(text or "").strip().lower()
    if not value:
        return "unknown"
    if "best" in value or "outstanding" in value or "award" in value:
        return "best_paper"
    if "findings" in value:
        return "findings"
    if "workshop" in value:
        return "workshop"
    if "demo" in value or "demonstration" in value:
        return "demo"
    if "industry" in value:
        return "industry"
    if "main" in value or "conference" in value or "long paper" in value or "short paper" in value:
        return "main_paper"
    return "unknown"


def is_accepted_main_conference(record: dict) -> bool:
    """Return True only for accepted main-conference papers."""
    blob = " ".join(str(record.get(key, "")) for key in ("decision", "venue", "venueid", "track", "paper_type")).lower()
    if any(term in blob for term in ("rejected", "withdrawn", "desk rejected", "workshop", "findings", "demo", "tutorial")):
        return False
    return any(term in blob for term in ("accept", "accepted", "main", "conference", "poster", "spotlight", "oral"))


def keep_by_presentation_type(record: dict, allowed: set[str]) -> bool:
    """Keep only allowed presentation types."""
    return normalize_presentation_type(record.get("presentation_type") or record.get("paper_type")) in allowed


def keep_nlp_paper(record: dict, allowed: set[str]) -> bool:
    """Keep NLP records by normalized paper_type."""
    paper_type = normalize_nlp_paper_type(record.get("paper_type"))
    return paper_type in allowed


def build_summary(records_seen: list[dict], records_kept: list[dict], source_domain: str, year: int) -> dict:
    """Build sidecar summary for output."""
    venues: dict[str, dict] = defaultdict(dict)
    seen_counter = Counter(record.get("venue") or "UNKNOWN" for record in records_seen)
    kept_by_venue: dict[str, list[dict]] = defaultdict(list)
    for record in records_kept:
        kept_by_venue[record.get("venue") or "UNKNOWN"].append(record)

    for venue, total_seen in seen_counter.items():
        kept = kept_by_venue.get(venue, [])
        presentation_counts = Counter(record.get("presentation_type") or "unknown" for record in kept)
        status_counts = Counter(record.get("topic_status") or "not_found" for record in kept)
        paper_type_counts = Counter(record.get("paper_type") or "unknown" for record in kept)
        venues[venue] = {
            "total_seen": total_seen,
            "total_kept": len(kept),
            **dict(sorted(presentation_counts.items())),
            **dict(sorted(paper_type_counts.items())),
            **dict(sorted(status_counts.items())),
        }

    return {
        "source_domain": source_domain,
        "year": year,
        "total_seen": len(records_seen),
        "total_kept": len(records_kept),
        "venues": dict(sorted(venues.items())),
    }


def group_records_by_venue(records: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        venue = (record.get("venue") or "unknown").lower()
        year = record.get("year") or "unknown"
        grouped[f"{venue}_{year}"].append(normalize_record(record))
    return dict(grouped)
