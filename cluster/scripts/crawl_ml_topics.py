#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topic_crawler.common import (  # noqa: E402
    build_summary,
    fetch_text,
    group_records_by_venue,
    read_json_or_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from topic_crawler.html_utils import parse_cfp_topics_from_html  # noqa: E402
from topic_crawler.openreview_utils import (  # noqa: E402
    extract_openreview_area,
    extract_openreview_content_value,
    extract_openreview_presentation_type,
    is_openreview_accepted,
    iter_openreview_notes,
)
from topic_crawler.schemas import normalize_record  # noqa: E402
from topic_crawler.sources import ML_SOURCES  # noqa: E402


ALLOWED_ML_PRESENTATION_TYPES = {"oral", "spotlight", "poster"}
LOGGER = logging.getLogger("crawl_ml_topics")


def build_record_from_note(note: dict, source: dict, cfp_topics: list[str]) -> dict:
    content = note.get("content") or {}
    title = extract_openreview_content_value(content, ["Title", "title"]) or ""
    authors = extract_openreview_content_value(content, ["Authors", "authors"]) or []
    if isinstance(authors, str):
        authors = [part.strip() for part in authors.split(",") if part.strip()]
    abstract = extract_openreview_content_value(content, ["Abstract", "abstract"]) or ""
    presentation_type = extract_openreview_presentation_type(note)
    area = extract_openreview_area(note)
    topic_status = "not_found"
    confidence = 0.0
    if area["official_category_raw"]:
        topic_status = "official_per_paper"
        confidence = 0.95
    elif area["official_keywords"]:
        topic_status = "official_keywords_only"
        confidence = 0.75
    elif cfp_topics:
        topic_status = "cfp_taxonomy_only"
        confidence = 0.45

    return normalize_record(
        {
            "paper_id": f"openreview:{note.get('forum') or note.get('id') or ''}",
            "source_domain": "ML",
            "venue": source["venue"],
            "year": source["year"],
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "paper_url": f"https://openreview.net/forum?id={note.get('forum') or note.get('id')}" if note.get("forum") or note.get("id") else "",
            "pdf_url": "",
            "paper_type": presentation_type,
            "presentation_type": presentation_type,
            "is_best_paper": False,
            "is_main_paper": True,
            **area,
            "cfp_topic_source_url": source.get("cfp_url"),
            "cfp_candidate_topics": cfp_topics,
            "track": "main",
            "topic_status": topic_status,
            "topic_confidence": confidence,
            "raw_source": {
                "openreview_forum": note.get("forum"),
                "openreview_invitation": note.get("invitation") or note.get("invitations"),
                "html_source": None,
            },
        }
    )


def merge_input_records(records: list[dict], args, sources: dict) -> tuple[list[dict], list[dict]]:
    seen = []
    kept = []
    for raw in records:
        venue_key = str(raw.get("venue_key") or raw.get("venue") or "").lower()
        source = sources.get(venue_key) or next((src for src in sources.values() if src["venue"].lower() == venue_key), None)
        source = source or {"venue": raw.get("venue", ""), "year": raw.get("year") or args.year}
        record = normalize_record(
            {
                **raw,
                "source_domain": raw.get("source_domain") or "ML",
                "venue": raw.get("venue") or source["venue"],
                "year": raw.get("year") or source["year"],
            }
        )
        seen.append(record)
        if record["presentation_type"] in ALLOWED_ML_PRESENTATION_TYPES:
            kept.append(record)
        elif args.keep_accepted_without_presentation and record["presentation_type"] == "unknown":
            record["notes"] = record["notes"] or "Accepted main-conference paper but presentation type is missing."
            kept.append(record)
    return seen, kept


def crawl_source(source: dict, args) -> tuple[list[dict], list[dict]]:
    cfp_topics = []
    cfp_html = fetch_text(source.get("cfp_url"), args.cache_dir, args.force)
    if cfp_html:
        cfp_topics = parse_cfp_topics_from_html(cfp_html)
    notes = iter_openreview_notes(source.get("openreview_venue_id"))
    seen = []
    kept = []
    for note in notes:
        if not is_openreview_accepted(note):
            continue
        record = build_record_from_note(note, source, cfp_topics)
        seen.append(record)
        if record["presentation_type"] in ALLOWED_ML_PRESENTATION_TYPES:
            kept.append(record)
        elif args.keep_accepted_without_presentation and record["presentation_type"] == "unknown":
            record["notes"] = "Accepted main-conference paper but presentation type is missing."
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
    parser.add_argument("--year", type=int, help="Optional year filter. Omit to crawl each venue's configured latest year.")
    parser.add_argument("--venues", nargs="+", default=["neurips", "icml", "iclr"])
    parser.add_argument("--out", default="data/outputs/ml_topics_latest.jsonl")
    parser.add_argument("--input")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--keep-accepted-without-presentation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    selected = {
        key: value
        for key, value in ML_SOURCES.items()
        if key in args.venues and (args.year is None or value["year"] == args.year)
    }
    all_seen = []
    all_kept = []
    for key, source in selected.items():
        LOGGER.info("Crawling %s %s", source["venue"], source["year"])
        seen, kept = crawl_source(source, args)
        LOGGER.info("%s: kept %s/%s", key, len(kept), len(seen))
        all_seen.extend(seen)
        all_kept.extend(kept)

    if args.input:
        seen, kept = merge_input_records(read_json_or_jsonl(args.input), args, ML_SOURCES)
        all_seen.extend(seen)
        all_kept.extend(kept)

    write_jsonl_atomic(all_kept, args.out)
    write_per_venue_json(all_kept, args.out, list(selected.values()))
    summary_year = args.year if args.year is not None else "latest"
    summary = build_summary(all_seen, all_kept, "ML", summary_year)
    summary["filters"] = {"allowed_presentation_types": sorted(ALLOWED_ML_PRESENTATION_TYPES), "allowed_nlp_paper_types": None}
    write_json_atomic(summary, Path(args.out).with_suffix(".summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
