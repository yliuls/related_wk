#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topic_crawler.common import build_summary, fetch_text, group_records_by_venue, normalize_nlp_paper_type, read_json_or_jsonl, write_json_atomic, write_jsonl_atomic  # noqa: E402
from topic_crawler.html_utils import parse_acl_anthology_volume, parse_arr_area_keywords, parse_best_paper_page, parse_cfp_topics_from_html  # noqa: E402
from topic_crawler.openreview_utils import extract_openreview_area, iter_openreview_notes  # noqa: E402
from topic_crawler.schemas import normalize_record  # noqa: E402
from topic_crawler.sources import NLP_SOURCES  # noqa: E402


DEFAULT_ALLOWED_NLP_PAPER_TYPES = {"best_paper", "main_paper"}
LOGGER = logging.getLogger("crawl_nlp_topics")
GENERIC_ARR_TERMS = {
    "analysis",
    "applications",
    "automatic evaluation",
    "biases",
    "datasets",
    "evaluation",
    "evaluation and metrics",
    "generation",
    "human evaluation",
    "metrics",
    "modeling",
    "multilingualism",
    "reasoning",
    "resources",
    "robustness",
    "safety",
    "transfer",
}


def allowed_types(args) -> set[str]:
    allowed = set(DEFAULT_ALLOWED_NLP_PAPER_TYPES)
    if args.include_findings:
        allowed.add("findings")
    if args.include_industry_track:
        allowed.add("industry")
    if args.include_demo:
        allowed.add("demo")
    return allowed


def apply_cfp_taxonomy(record: dict, source: dict, cfp_topics: list[str]) -> dict:
    topic_status = record.get("topic_status")
    if not topic_status or topic_status == "not_found":
        topic_status = "cfp_taxonomy_only" if cfp_topics else "not_found"
    return normalize_record(
        {
            **record,
            "source_domain": "NLP",
            "venue": record.get("venue") or source["venue"],
            "year": record.get("year") or source["year"],
            "official_category_source": record.get("official_category_source") or ("cfp_topic_list" if cfp_topics else "none"),
            "cfp_topic_source_url": source.get("cfp_url"),
            "cfp_candidate_topics": record.get("cfp_candidate_topics") or cfp_topics,
            "topic_status": topic_status,
            "topic_confidence": record.get("topic_confidence") or (0.45 if cfp_topics else 0.0),
        }
    )


def apply_arr_area_inference(record: dict, arr_areas: list[dict]) -> dict:
    if record.get("official_category_raw") or not arr_areas:
        return normalize_record(record)
    match = infer_arr_area(record, arr_areas)
    if not match:
        return normalize_record(record)
    raw_source = dict(record.get("raw_source") or {})
    raw_source["arr_area_match_score"] = match["score"]
    raw_source["arr_area_keywords_url"] = match["source_url"]
    return normalize_record(
        {
            **record,
            "official_category_source": "arr_area_keywords_inferred",
            "official_category_raw": match["area"],
            "official_primary_area": match["area"],
            "official_secondary_area": None,
            "official_keywords": match["matched_keywords"],
            "track": match["area"],
            "topic_status": "arr_keyword_inferred",
            "topic_confidence": match["confidence"],
            "raw_source": raw_source,
        }
    )


def infer_arr_area(record: dict, arr_areas: list[dict]) -> dict | None:
    text = _normalize_match_text(" ".join([record.get("title") or "", record.get("abstract") or ""]))
    if not text:
        return None
    best = None
    for area in arr_areas:
        score = 0.0
        matched = []
        terms = [area["area"], *area.get("keywords", [])]
        for term in terms:
            normalized = _normalize_match_text(term)
            if not normalized or len(normalized) < 4:
                continue
            words = normalized.split()
            if normalized in GENERIC_ARR_TERMS:
                continue
            if _contains_term(text, normalized):
                weight = 2.0 + min(len(words), 4)
                if term == area["area"]:
                    weight += 2.0
                score += weight
                matched.append(term)
        if score and (best is None or score > best["score"]):
            best = {
                "area": area["area"],
                "score": score,
                "matched_keywords": matched[:8],
                "source_url": area.get("source_url"),
            }
    if not best or best["score"] < 5.0:
        return None
    best["confidence"] = min(0.75, 0.45 + best["score"] / 60.0)
    return best


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _contains_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


def openreview_records(source: dict, cfp_topics: list[str]) -> list[dict]:
    if not source.get("openreview_venue_id"):
        return []
    records = []
    for note in iter_openreview_notes(source["openreview_venue_id"]):
        content = note.get("content") or {}
        title = _content_value(content, ["Title", "title"]) or ""
        authors = _content_value(content, ["Authors", "authors"]) or []
        if isinstance(authors, str):
            authors = [part.strip() for part in authors.split(",") if part.strip()]
        area = extract_openreview_area(note)
        status = "official_per_paper" if area["official_category_raw"] else ("official_keywords_only" if area["official_keywords"] else "cfp_taxonomy_only")
        records.append(
            apply_cfp_taxonomy(
                {
                    "paper_id": f"openreview:{note.get('forum') or note.get('id') or ''}",
                    "title": title,
                    "authors": authors,
                    "paper_type": "main_paper",
                    "is_best_paper": False,
                    "is_main_paper": True,
                    **area,
                    "topic_status": status,
                    "raw_source": {"openreview_forum": note.get("forum"), "openreview_invitation": note.get("invitation")},
                },
                source,
                cfp_topics,
            )
        )
    return records


def _content_value(content: dict, keys: list[str]):
    lowered = {str(key).lower(): key for key in content}
    for key in keys:
        actual = key if key in content else lowered.get(key.lower())
        if actual is None:
            continue
        value = content[actual]
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if value not in (None, "", []):
            return value
    return None


def merge_best_flags(records: list[dict], best_records: list[dict]) -> list[dict]:
    best_titles = {record["title"].strip().lower() for record in best_records if record.get("title")}
    merged = []
    for record in records:
        if record.get("title", "").strip().lower() in best_titles:
            record = {**record, "paper_type": "best_paper", "is_best_paper": True, "is_main_paper": True}
        merged.append(normalize_record(record))
    seen_titles = {record.get("title", "").strip().lower() for record in merged}
    for record in best_records:
        title_key = record.get("title", "").strip().lower()
        if title_key and title_key not in seen_titles:
            merged.append(normalize_record(record))
    return merged


def crawl_source(source: dict, args) -> tuple[list[dict], list[dict]]:
    cfp_html = fetch_text(source.get("cfp_url"), args.cache_dir, args.force)
    cfp_topics = parse_cfp_topics_from_html(cfp_html) if cfp_html else []
    arr_html = fetch_text(source.get("arr_area_keywords_url"), args.cache_dir, args.force)
    arr_areas = parse_arr_area_keywords(arr_html) if arr_html else []
    for area in arr_areas:
        area["source_url"] = source.get("arr_area_keywords_url")
    anthology_html = fetch_text(source.get("anthology_url"), args.cache_dir, args.force)
    records = parse_acl_anthology_volume(anthology_html, source["venue"], source["year"]) if anthology_html else []
    awards_html = fetch_text(source.get("awards_url"), args.cache_dir, args.force)
    best_records = parse_best_paper_page(awards_html, source["venue"], source["year"]) if awards_html else []
    records = [apply_arr_area_inference(apply_cfp_taxonomy(record, source, cfp_topics), arr_areas) for record in records]
    records.extend(openreview_records(source, cfp_topics))
    records = merge_best_flags(records, [apply_arr_area_inference(apply_cfp_taxonomy(record, source, cfp_topics), arr_areas) for record in best_records])
    allowed = allowed_types(args)
    seen = records
    kept = [record for record in records if normalize_nlp_paper_type(record["paper_type"]) in allowed]
    return seen, kept


def merge_input_records(records: list[dict], args) -> tuple[list[dict], list[dict]]:
    seen = []
    kept = []
    allowed = allowed_types(args)
    for raw in records:
        venue_key = str(raw.get("venue_key") or raw.get("venue") or "").lower()
        source = NLP_SOURCES.get(venue_key) or next((src for src in NLP_SOURCES.values() if src["venue"].lower() == venue_key), None)
        source = source or {"venue": raw.get("venue", ""), "year": raw.get("year") or args.year, "cfp_url": None}
        record = apply_cfp_taxonomy(raw, source, raw.get("cfp_candidate_topics") or [])
        record["paper_type"] = normalize_nlp_paper_type(record.get("paper_type"))
        if record["paper_type"] == "unknown":
            record["paper_type"] = "main_paper"
        record = normalize_record(record)
        seen.append(record)
        if record["paper_type"] in allowed:
            kept.append(record)
    return seen, kept


def write_per_venue_json(records: list[dict], out_path: str, sources: list[dict] | None = None) -> None:
    out_dir = Path(out_path).parent
    grouped = group_records_by_venue(records)
    for source in sources or []:
        grouped.setdefault(f"{source['venue'].lower()}_{source['year']}", [])
    for name, group in grouped.items():
        write_json_atomic(group, out_dir / f"{name}.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--venues", nargs="+", default=["acl", "emnlp", "naacl"])
    parser.add_argument("--out", default="data/outputs/nlp_topics_2025.jsonl")
    parser.add_argument("--input")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--include-findings", action="store_true")
    parser.add_argument("--include-industry-track", action="store_true")
    parser.add_argument("--include-demo", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    selected = {key: value for key, value in NLP_SOURCES.items() if key in args.venues and value["year"] == args.year}
    all_seen = []
    all_kept = []
    for key, source in selected.items():
        LOGGER.info("Crawling %s %s", source["venue"], source["year"])
        seen, kept = crawl_source(source, args)
        LOGGER.info("%s: kept %s/%s", key, len(kept), len(seen))
        all_seen.extend(seen)
        all_kept.extend(kept)

    if args.input:
        seen, kept = merge_input_records(read_json_or_jsonl(args.input), args)
        all_seen.extend(seen)
        all_kept.extend(kept)

    write_jsonl_atomic(all_kept, args.out)
    write_per_venue_json(all_kept, args.out, list(selected.values()))
    summary = build_summary(all_seen, all_kept, "NLP", args.year)
    summary["filters"] = {"allowed_presentation_types": None, "allowed_nlp_paper_types": sorted(allowed_types(args))}
    write_json_atomic(summary, Path(args.out).with_suffix(".summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
