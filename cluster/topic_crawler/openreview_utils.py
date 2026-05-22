from __future__ import annotations

import inspect
import logging

from .common import normalize_presentation_type, normalize_topic_text, split_hierarchical_topic


LOGGER = logging.getLogger(__name__)

AREA_KEYS = [
    "Primary Area",
    "primary_area",
    "Subject Area",
    "subject_area",
    "Area",
    "area",
    "Track",
    "track",
]
KEYWORD_KEYS = ["Keywords", "keywords", "Keyword", "keyword"]
DECISION_KEYS = ["Decision", "decision", "recommendation", "Recommendation"]
PRESENTATION_KEYS = ["Presentation Type", "presentation_type", "Presentation", "presentation", "Venue", "venue"]


def get_openreview_client():
    """Create OpenReview client. Prefer OpenReview API v2 when possible."""
    try:
        import openreview

        if hasattr(openreview, "api") and hasattr(openreview.api, "OpenReviewClient"):
            return openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
        if hasattr(openreview, "Client"):
            return openreview.Client(baseurl="https://api.openreview.net")
    except Exception as exc:
        LOGGER.warning("OpenReview client unavailable: %s", exc)
    return None


def iter_openreview_notes(venue_id: str, limit: int = 1000) -> list[dict]:
    """Iterate OpenReview notes with pagination."""
    client = get_openreview_client()
    if client is None or not venue_id:
        return []
    notes = []
    try:
        if hasattr(client, "get_all_notes"):
            notes = _call_openreview_notes(client.get_all_notes, content={"venueid": venue_id}, limit=limit)
            if not notes:
                notes = _call_openreview_notes(client.get_all_notes, invitation=f"{venue_id}/-/Submission", limit=limit)
        else:
            notes = client.get_notes(content={"venueid": venue_id}, limit=limit)
    except Exception as exc:
        LOGGER.warning("OpenReview query failed for %s: %s", venue_id, exc)
        return []
    return [note_to_dict(note) for note in notes]


def _call_openreview_notes(method, *, limit: int, **kwargs):
    """Call OpenReview note methods across client versions."""
    signature = inspect.signature(method)
    if "limit" in signature.parameters:
        kwargs["limit"] = limit
    return method(**kwargs)


def note_to_dict(note) -> dict:
    if isinstance(note, dict):
        return note
    data = {}
    for field in ("id", "forum", "number", "content", "invitation", "invitations", "venue", "venueid"):
        if hasattr(note, field):
            data[field] = getattr(note, field)
    return data


def extract_openreview_content_value(content: dict, candidate_keys: list[str]):
    """Extract value from OpenReview content dict, supporting {'value': ...} and raw formats."""
    content = content or {}
    lowered = {str(key).lower(): key for key in content}
    for key in candidate_keys:
        actual_key = key if key in content else lowered.get(key.lower())
        if actual_key is None:
            continue
        value = content.get(actual_key)
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if value not in (None, "", []):
            return value
    return None


def extract_openreview_area(note: dict) -> dict:
    """Extract Primary Area / Subject Area / Area / Track / Keywords from a note."""
    content = note.get("content") or {}
    raw_area = extract_openreview_content_value(content, AREA_KEYS)
    keywords = extract_openreview_content_value(content, KEYWORD_KEYS)
    if isinstance(keywords, str):
        keywords = [part.strip() for part in keywords.replace(";", ",").split(",") if part.strip()]
    elif keywords is None:
        keywords = []

    category_source = "none"
    for key in AREA_KEYS:
        if extract_openreview_content_value(content, [key]):
            category_source = {
                "Primary Area": "openreview_primary_area",
                "primary_area": "openreview_primary_area",
                "Subject Area": "openreview_subject_area",
                "subject_area": "openreview_subject_area",
                "Area": "openreview_area",
                "area": "openreview_area",
                "Track": "openreview_track",
                "track": "openreview_track",
            }.get(key, "openreview_area")
            break
    primary, secondary = split_hierarchical_topic(raw_area)
    return {
        "official_category_source": category_source if raw_area else ("openreview_keywords" if keywords else "none"),
        "official_category_raw": normalize_topic_text(raw_area),
        "official_primary_area": primary,
        "official_secondary_area": secondary,
        "official_keywords": keywords or [],
    }


def extract_openreview_presentation_type(note: dict) -> str:
    """Extract and normalize oral / spotlight / poster if available."""
    content = note.get("content") or {}
    value = extract_openreview_content_value(content, PRESENTATION_KEYS)
    if not value:
        value = note.get("venue") or note.get("venueid") or ""
    return normalize_presentation_type(value)


def is_openreview_accepted(note: dict) -> bool:
    """Infer accepted main-conference status from decision, invitation, venue, venueid, or content fields."""
    content = note.get("content") or {}
    decision = extract_openreview_content_value(content, DECISION_KEYS)
    blob = " ".join(
        str(part)
        for part in [
            decision,
            note.get("venue"),
            note.get("venueid"),
            note.get("invitation"),
            " ".join(note.get("invitations") or []),
        ]
        if part
    ).lower()
    if any(term in blob for term in ("reject", "withdraw", "desk reject", "workshop", "retracted")):
        return False
    return any(term in blob for term in ("accept", "poster", "spotlight", "oral", "conference"))
