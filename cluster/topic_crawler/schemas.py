from __future__ import annotations

from copy import deepcopy


STANDARD_FIELDS = [
    "title",
    "paper_id",
    "source_domain",
    "venue",
    "year",
    "authors",
    "abstract",
    "paper_url",
    "pdf_url",
    "paper_type",
    "presentation_type",
    "is_best_paper",
    "is_main_paper",
    "official_category_source",
    "official_category_raw",
    "official_primary_area",
    "official_secondary_area",
    "official_keywords",
    "cfp_topic_source_url",
    "cfp_candidate_topics",
    "poster_session",
    "poster_number",
    "track",
    "topic_status",
    "topic_confidence",
    "notes",
    "raw_source",
]

LIST_FIELDS = {"authors", "official_keywords", "cfp_candidate_topics"}
DICT_FIELDS = {"raw_source"}
BOOL_FIELDS = {"is_best_paper", "is_main_paper"}
INT_FIELDS = {"year"}
FLOAT_FIELDS = {"topic_confidence"}


DEFAULT_RECORD = {
    "title": "",
    "paper_id": "",
    "source_domain": "",
    "venue": "",
    "year": None,
    "authors": [],
    "abstract": "",
    "paper_url": "",
    "pdf_url": "",
    "paper_type": "unknown",
    "presentation_type": "unknown",
    "is_best_paper": False,
    "is_main_paper": True,
    "official_category_source": "none",
    "official_category_raw": None,
    "official_primary_area": None,
    "official_secondary_area": None,
    "official_keywords": [],
    "cfp_topic_source_url": None,
    "cfp_candidate_topics": [],
    "poster_session": None,
    "poster_number": None,
    "track": "main",
    "topic_status": "not_found",
    "topic_confidence": 0.0,
    "notes": "",
    "raw_source": {},
}


def make_empty_record() -> dict:
    """Return a record with all standard fields initialized."""
    return deepcopy(DEFAULT_RECORD)


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_record(record: dict) -> dict:
    """Fill missing fields and normalize data types."""
    normalized = make_empty_record()
    normalized.update(record or {})

    for field in LIST_FIELDS:
        normalized[field] = _as_list(normalized.get(field))
    for field in DICT_FIELDS:
        if not isinstance(normalized.get(field), dict):
            normalized[field] = {}
    for field in BOOL_FIELDS:
        normalized[field] = bool(normalized.get(field))
    for field in INT_FIELDS:
        value = normalized.get(field)
        if value not in (None, ""):
            normalized[field] = int(value)
    for field in FLOAT_FIELDS:
        value = normalized.get(field)
        normalized[field] = float(value or 0.0)

    if normalized["is_best_paper"]:
        normalized["paper_type"] = "best_paper"
        normalized["is_main_paper"] = True

    return {field: normalized.get(field) for field in STANDARD_FIELDS}


def validate_record(record: dict) -> tuple[bool, list[str]]:
    """Validate required fields and return errors."""
    errors = []
    missing = [field for field in STANDARD_FIELDS if field not in record]
    if missing:
        errors.append(f"Missing standard fields: {', '.join(missing)}")

    normalized = normalize_record(record)
    for field in ("source_domain", "venue", "title"):
        if not normalized.get(field):
            errors.append(f"Missing required field: {field}")
    if normalized.get("year") is None:
        errors.append("Missing required field: year")
    if not isinstance(normalized.get("authors"), list):
        errors.append("authors must be a list")
    if not isinstance(normalized.get("raw_source"), dict):
        errors.append("raw_source must be a dict")

    allowed_topic_status = {
        "official_per_paper",
        "official_keywords_only",
        "arr_keyword_inferred",
        "cfp_taxonomy_only",
        "presentation_metadata_only",
        "not_found",
    }
    if normalized.get("topic_status") not in allowed_topic_status:
        errors.append(f"Invalid topic_status: {normalized.get('topic_status')}")

    return not errors, errors
