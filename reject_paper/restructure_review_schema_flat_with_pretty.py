#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restructure OpenReview-style JSONL into a compact review/rebuttal/AC schema.

Usage:
  # JSONL output, one paper per line
  python restructure_review_schema_flat_with_pretty.py -i raw_openreview.jsonl -o output.jsonl

  # Pretty JSON output, easier for manual inspection
  python restructure_review_schema_flat_with_pretty.py -i raw_openreview.jsonl -o output_pretty.json --pretty
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METADATA_KEYS = {
    "submission_id",
    "submission_number",
    "submission_title",
    "reply_id",
    "reply_type",
    "reviewer_id",
    "forum",
    "replyto",
    "signatures",
    "readers",
    "cdate",
    "mdate",
}

PAPER_KEYS = [
    "paper_id",
    "forum_id",
    "number",
    "url",
    "title",
    "abstract",
    "TLDR",
    "keywords",
    "primary_area",
    "venue",
    "venueid",
    "paperhash",
]

COUNT_KEYS = [
    "num_replies",
    "num_official_reviews",
    "num_reviewer_comments",
    "num_area_chair_summaries",
    "num_area_chair_comments",
    "num_decisions",
    "num_public_comments",
    "num_other_replies",
]


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None

    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r"\bnull\b", "None", line)
    fixed = re.sub(r"\btrue\b", "True", fixed)
    fixed = re.sub(r"\bfalse\b", "False", fixed)
    try:
        obj = ast.literal_eval(fixed)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(obj, dict):
            yield obj
            return
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        record = safe_json_loads(line)
        if record is not None:
            yield record


def compact_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: item.get(k) for k in METADATA_KEYS if k in item}


def reviewer_short_id(reviewer_id: str) -> str:
    if not reviewer_id:
        return ""
    return reviewer_id.replace("Reviewer_", "").replace("Reviewer ", "").strip()


def reply_author_type(item: Dict[str, Any]) -> str:
    signatures = item.get("signatures") or []
    sig_text = " ".join(map(str, signatures))
    if "Reviewer_" in sig_text:
        return "reviewer"
    if "Authors" in sig_text:
        return "authors"
    if "Area_Chair" in sig_text or "Area" in sig_text:
        return "area_chair"
    if "Program_Chairs" in sig_text:
        return "program_chairs"
    return "unknown"


def normalize_reply(item: Dict[str, Any]) -> Dict[str, Any]:
    content = item.get("content") or {}
    out = {"metadata": compact_metadata(item)}
    if isinstance(content, dict):
        out.update(content)
    else:
        out["text"] = str(content)
    return out


def normalize_official_review(review: Dict[str, Any]) -> Dict[str, Any]:
    content = review.get("content") or {}
    out = {"metadata": compact_metadata(review)}
    if isinstance(content, dict):
        out.update(content)
    else:
        out["text"] = str(content)
    return out


def collect_author_rebuttals_for_review(record: Dict[str, Any], review_reply_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in record.get("public_comments") or []:
        if reply_author_type(c) != "authors":
            continue
        if c.get("replyto") == review_reply_id:
            out.append(normalize_reply(c))
    return sorted(out, key=lambda x: x.get("metadata", {}).get("cdate") or 0)


def collect_reviewer_comments_for_review(
    record: Dict[str, Any],
    reviewer_id: str,
    review_reply_id: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in record.get("reviewer_comments") or []:
        same_reviewer = c.get("reviewer_id") == reviewer_id
        same_thread = c.get("replyto") == review_reply_id
        if same_reviewer or same_thread:
            out.append(normalize_reply(c))
    return sorted(out, key=lambda x: x.get("metadata", {}).get("cdate") or 0)


def collect_global_author_rebuttals(record: Dict[str, Any], review_reply_ids: set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in record.get("public_comments") or []:
        if reply_author_type(c) != "authors":
            continue
        if c.get("replyto") not in review_reply_ids:
            out.append(normalize_reply(c))
    return sorted(out, key=lambda x: x.get("metadata", {}).get("cdate") or 0)


def normalize_area_chair_evaluation(record: Dict[str, Any]) -> Dict[str, Any]:
    ac_summaries = record.get("area_chair_summaries") or []
    keep_keys = ["summary", "reviewer_concerns", "reviewer_scores"]
    merged: Dict[str, Any] = {}

    for key in keep_keys:
        values = []
        for s in ac_summaries:
            content = s.get("content") or {}
            value = content.get(key)
            if value not in (None, ""):
                values.append(str(value))
        merged[key] = "\n\n---\n\n".join(values) if values else ""

    metadata = [compact_metadata(s) for s in ac_summaries]
    if metadata:
        merged["metadata"] = metadata

    return merged


def normalize_final_decision(record: Dict[str, Any]) -> Dict[str, Any]:
    decisions = record.get("decisions") or []
    if not decisions:
        return {}

    d = sorted(decisions, key=lambda x: (x.get("cdate") or 0, x.get("mdate") or 0))[-1]
    content = d.get("content") or {}

    out = {"metadata": compact_metadata(d)}
    if isinstance(content, dict):
        out.update(content)
    else:
        out["text"] = str(content)
    return out


def restructure_record(record: Dict[str, Any]) -> Dict[str, Any]:
    official_reviews = record.get("official_reviews") or []
    review_reply_ids = {r.get("reply_id") for r in official_reviews if r.get("reply_id")}

    reviews = []
    for r in official_reviews:
        reviewer_id = r.get("reviewer_id") or ""
        reply_id = r.get("reply_id") or ""

        reviews.append({
            "reviewer_id": reviewer_id,
            "reviewer_id_short": reviewer_short_id(reviewer_id),
            "official_review": normalize_official_review(r),
            "author_rebuttals_to_this_reviewer": collect_author_rebuttals_for_review(record, reply_id),
            "reviewer_post_rebuttal_comments": collect_reviewer_comments_for_review(record, reviewer_id, reply_id),
        })

    return {
        "schema_version": "review_rebuttal_ac_flat_v1",
        "paper": {k: record.get(k) for k in PAPER_KEYS if k in record},
        "reviews": reviews,
        "area_chair_evaluation": normalize_area_chair_evaluation(record),
        "global_author_rebuttals": collect_global_author_rebuttals(record, review_reply_ids),
        "final_decision": normalize_final_decision(record),
        "counts": {k: record.get(k) for k in COUNT_KEYS if k in record},
    }


def write_output(items: List[Dict[str, Any]], output_path: Path, pretty: bool) -> None:
    if pretty:
        output_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        with output_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input JSONL/JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL/JSON file")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Output a pretty JSON array instead of JSONL",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = list(iter_jsonl_records(input_path))
    transformed = [restructure_record(r) for r in records]
    write_output(transformed, output_path, pretty=args.pretty)

    print(f"Loaded {len(records)} records")
    print(f"Wrote {len(transformed)} records to {output_path}")
    print(f"Format: {'pretty JSON array' if args.pretty else 'JSONL'}")


if __name__ == "__main__":
    main()
