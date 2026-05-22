#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topic_crawler.common import build_summary, fetch_text, group_records_by_venue, read_json_or_jsonl, write_json_atomic, write_jsonl_atomic  # noqa: E402
from topic_crawler.html_utils import parse_cfp_topics_from_html, parse_cvf_accepted_papers, parse_cvf_paper_detail  # noqa: E402
from topic_crawler.schemas import normalize_record  # noqa: E402
from topic_crawler.sources import CV_SOURCES  # noqa: E402

try:
    import httpx
except ImportError:  # pragma: no cover - optional fallback transport
    httpx = None


ALLOWED_CV_PRESENTATION_TYPES = {"oral", "spotlight", "poster"}
LOGGER = logging.getLogger("crawl_cv_topics")
CVF_BASE_URL = "https://openaccess.thecvf.com"


def enrich_cv_record(record: dict, source: dict, cfp_topics: list[str]) -> dict:
    return normalize_record(
        {
            **record,
            "source_domain": "CV",
            "venue": record.get("venue") or source["venue"],
            "year": record.get("year") or source["year"],
            "official_category_source": "cfp_topic_list" if cfp_topics else "none",
            "official_category_raw": None,
            "official_primary_area": None,
            "official_secondary_area": None,
            "official_keywords": [],
            "cfp_topic_source_url": source.get("cfp_url"),
            "cfp_candidate_topics": cfp_topics,
            "topic_status": record.get("topic_status") or "presentation_metadata_only",
            "topic_confidence": record.get("topic_confidence") or 0.35,
            "notes": record.get("notes") or "CV public page does not expose stable per-paper official topic; poster session is metadata, not topic.",
        }
    )


def fetch_cvf_detail_html(url: str, args) -> str | None:
    html = fetch_text(
        url,
        args.cache_dir,
        args.force,
        timeout=args.abstract_timeout,
        max_retries=args.abstract_retries,
    )
    if html or httpx is None:
        return html

    try:
        response = httpx.get(
            url,
            timeout=args.abstract_timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if response.status_code in (403, 404):
            LOGGER.warning("Fetch skipped for %s: HTTP %s", url, response.status_code)
            return None
        response.raise_for_status()
        html = response.text
        if args.cache_dir:
            cache_path = Path(args.cache_dir) / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.html"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html, encoding="utf-8")
        return html
    except Exception as exc:
        LOGGER.warning("HTTPX fallback failed for %s: %s", url, exc)
        return None


def enrich_cvf_abstract(record: dict, args) -> dict:
    paper_url = record.get("paper_url") or ""
    if record.get("abstract") or not paper_url:
        return record
    detail_url = urljoin(CVF_BASE_URL, paper_url)
    detail_html = fetch_cvf_detail_html(detail_url, args)
    detail = parse_cvf_paper_detail(detail_html) if detail_html else {}
    if not detail:
        return record

    enriched = {**record}
    if detail.get("abstract"):
        enriched["abstract"] = detail["abstract"]
    if detail.get("pdf_url"):
        enriched["pdf_url"] = detail["pdf_url"]
    if detail.get("authors"):
        enriched["authors"] = detail["authors"]
    raw_source = dict(enriched.get("raw_source") or {})
    raw_source["paper_detail_url"] = detail_url
    enriched["raw_source"] = raw_source
    return enriched


def enrich_cvf_abstracts(records: list[dict], args) -> list[dict]:
    if not records or args.abstract_workers <= 1:
        return [enrich_cvf_abstract(record, args) for record in records]

    enriched = [None] * len(records)
    with ThreadPoolExecutor(max_workers=args.abstract_workers) as executor:
        future_to_index = {
            executor.submit(enrich_cvf_abstract, record, args): index
            for index, record in enumerate(records)
        }
        for completed, future in enumerate(as_completed(future_to_index), start=1):
            index = future_to_index[future]
            try:
                enriched[index] = future.result()
            except Exception as exc:
                LOGGER.warning("Failed to enrich CVF abstract for %s: %s", records[index].get("title"), exc)
                enriched[index] = records[index]
            if completed % 250 == 0:
                LOGGER.info("Fetched CVF detail abstracts: %s/%s", completed, len(records))
    return enriched


def crawl_source(source: dict, args) -> tuple[list[dict], list[dict]]:
    accepted_html = fetch_text(source.get("accepted_url"), args.cache_dir, args.force, timeout=args.timeout)
    raw_records = parse_cvf_accepted_papers(accepted_html, source["venue"], source["year"]) if accepted_html else []
    raw_records = enrich_cvf_abstracts(raw_records, args)
    cfp_html = fetch_text(source.get("cfp_url"), args.cache_dir, args.force, timeout=args.timeout)
    cfp_topics = parse_cfp_topics_from_html(cfp_html) if cfp_html else []
    seen = [enrich_cv_record(record, source, cfp_topics) for record in raw_records]
    kept = [record for record in seen if record["presentation_type"] in ALLOWED_CV_PRESENTATION_TYPES]
    return seen, kept


def merge_input_records(records: list[dict], args) -> tuple[list[dict], list[dict]]:
    seen = []
    kept = []
    for raw in records:
        venue_key = str(raw.get("venue_key") or raw.get("venue") or "").lower()
        source = CV_SOURCES.get(venue_key) or next((src for src in CV_SOURCES.values() if src["venue"].lower() == venue_key), None)
        source = source or {"venue": raw.get("venue", ""), "year": raw.get("year") or args.year, "cfp_url": None}
        record = enrich_cv_record(raw, source, raw.get("cfp_candidate_topics") or [])
        seen.append(record)
        if record["presentation_type"] in ALLOWED_CV_PRESENTATION_TYPES:
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
    parser.add_argument("--venues", nargs="+", default=["cvpr", "iccv"])
    parser.add_argument("--include-eccv-latest", action="store_true")
    parser.add_argument("--out", default="data/outputs/cv_topics_2025.jsonl")
    parser.add_argument("--input")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--abstract-workers", type=int, default=12)
    parser.add_argument("--abstract-timeout", type=int, default=20)
    parser.add_argument("--abstract-retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    venue_keys = set(args.venues)
    if args.include_eccv_latest:
        venue_keys.add("eccv")
    selected = {key: value for key, value in CV_SOURCES.items() if key in venue_keys and (value["year"] == args.year or key == "eccv")}
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
    summary = build_summary(all_seen, all_kept, "CV", args.year)
    summary["filters"] = {"allowed_presentation_types": sorted(ALLOWED_CV_PRESENTATION_TYPES), "allowed_nlp_paper_types": None}
    write_json_atomic(summary, Path(args.out).with_suffix(".summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
