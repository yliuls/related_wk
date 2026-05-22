#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map ICLR 2026 rejected paper topics to unified canonical topics.

Input:
  A JSONL file, e.g. iclr2026_rejected_topics.jsonl.
  Each line should be one paper record and should contain:
    - official_primary_area
  Optional fields:
    - venue
    - title
    - abstract
    - paper_id

Output:
  A JSONL file where each original record is preserved, paper_id is reset
  to 00001, 00002, ... in output order, and these fields are added:
    - original_paper_id
    - canonical_topic
    - canonical_topic_before_merge
    - needs_llm_refinement
    - mapping_confidence
    - mapping_reason
    - matched_official_primary_area
    - mapping_score
    - llm_candidate_topics

Main rule:
  ICLR official_primary_area is first mapped like map_canonical_topics.py.
  Then:
    - "Other topics in machine learning"
    - "General machine learning"
  are merged into:
    - "Machine Learning: General, Advanced & Cross-cutting Topics"

Example:
  python map_iclr2026_rejected_topics.py \
    --input iclr2026_rejected_topics.jsonl \
    --output iclr2026_rejected_topics_mapped.jsonl \
    --summary-output iclr2026_rejected_topics_mapped_summary.json \
    --llm-todo-output iclr2026_rejected_topics_need_review.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
import difflib
import json
import re
from pathlib import Path
from typing import Any


MERGED_TOPIC_L2 = "Machine Learning: General, Advanced & Cross-cutting Topics"

# Final topic list after merging General ML and Other ML.
FINAL_CANONICAL_TOPICS = [
    "Unsupervised, self-supervised, semi-supervised, and supervised representation learning",
    "Transfer learning, meta learning, and lifelong learning",
    "Reinforcement learning",
    "Applications to computer vision, audio, language, and other modalities",
    "Metric learning, kernel learning",
    "Probabilistic methods",
    "Generative models",
    "Foundation or frontier models, including LLMs",
    "Causal reasoning",
    "Optimization",
    "Theory of machine learning",
    "Learning on graphs and other geometries & topologies",
    "Learning on time series and dynamical systems",
    "Societal considerations including fairness, safety, privacy",
    "Interpretability and explainable AI",
    "Datasets, benchmarks, and evaluation",
    "Infrastructure, software libraries, hardware, systems, etc.",
    "Neurosymbolic & hybrid AI systems",
    "Applications to robotics, autonomy, planning",
    "Applications to neuroscience & cognitive science",
    "Applications to physical sciences, life sciences, and earth sciences",
    "Applications to health, medicine, sustainability, and social sciences",
    MERGED_TOPIC_L2,
]

# This keeps the first script's ICLR mapping behavior.
# The merge is applied after this raw mapping.
ICLR_PRIMARY_TO_CANONICAL_RAW = {
    "unsupervised, self-supervised, semi-supervised, and supervised representation learning":
        "Unsupervised, self-supervised, semi-supervised, and supervised representation learning",

    "transfer learning, meta learning, and lifelong learning":
        "Transfer learning, meta learning, and lifelong learning",

    "reinforcement learning":
        "Reinforcement learning",

    "representation learning for computer vision, audio, language, and other modalities":
        "Applications to computer vision, audio, language, and other modalities",

    "applications to computer vision, audio, language, and other modalities":
        "Applications to computer vision, audio, language, and other modalities",

    "metric learning, kernel learning":
        "Metric learning, kernel learning",

    "probabilistic methods (Bayesian methods, variational inference, sampling, UQ, etc.)":
        "Probabilistic methods",

    "generative models":
        "Generative models",

    "foundation or frontier models, including LLMs":
        "Foundation or frontier models, including LLMs",

    "causal reasoning":
        "Causal reasoning",

    "optimization":
        "Optimization",

    "learning theory":
        "Theory of machine learning",

    "learning on graphs and other geometries & topologies":
        "Learning on graphs and other geometries & topologies",

    "learning on time series and dynamical systems":
        "Learning on time series and dynamical systems",

    "societal considerations including fairness, safety, privacy":
        "Societal considerations including fairness, safety, privacy",

    "alignment, fairness, safety, privacy, and societal considerations":
        "Societal considerations including fairness, safety, privacy",

    "visualization or interpretation of learned representations":
        "Interpretability and explainable AI",

    "interpretability and explainable AI":
        "Interpretability and explainable AI",

    "datasets and benchmarks":
        "Datasets, benchmarks, and evaluation",

    "infrastructure, software libraries, hardware, etc.":
        "Infrastructure, software libraries, hardware, systems, etc.",

    "infrastructure, software libraries, hardware, systems, etc.":
        "Infrastructure, software libraries, hardware, systems, etc.",

    "neurosymbolic & hybrid AI systems (physics-informed, logic & formal reasoning, etc.)":
        "Neurosymbolic & hybrid AI systems",

    "applications to robotics, autonomy, planning":
        "Applications to robotics, autonomy, planning",

    "applications to neuroscience & cognitive science":
        "Applications to neuroscience & cognitive science",

    "applications to physical sciences (physics, chemistry, biology, etc.)":
        "Applications to physical sciences, life sciences, and earth sciences",

    # These two are intentionally kept as pre-merge topics here.
    # They will be merged by merge_general_and_other_ml().
    "general machine learning (i.e., none of the above)":
        "General machine learning",

    "other topics in machine learning (i.e., none of the above)":
        "Other topics in machine learning",
}


def normalize_text(x: Any) -> str:
    """Normalize topic strings for exact/fuzzy matching."""
    if x is None:
        return ""

    s = str(x).strip().lower()
    if s in {"", "none", "null", "nan"}:
        return ""

    s = s.replace("_", " ")
    s = s.replace("&", "and")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^\w\s,+()/.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_venue(x: Any, default: str = "ICLR") -> str:
    s = normalize_text(x)
    if not s:
        return default

    if s == "iclr":
        return "ICLR"
    if s == "icml":
        return "ICML"
    if s in {"neurips", "nips"}:
        return "NeurIPS"

    return str(x).strip()


def build_normalized_map(raw_map: dict[str, str]) -> dict[str, str]:
    return {normalize_text(k): v for k, v in raw_map.items()}


ICLR_PRIMARY_TO_CANONICAL = build_normalized_map(ICLR_PRIMARY_TO_CANONICAL_RAW)


def fuzzy_lookup(
    key: str,
    mapping: dict[str, str],
    threshold: float = 0.86,
) -> tuple[str | None, str | None, float]:
    """
    Return:
      mapped_value, matched_key, score

    If no good fuzzy match:
      None, None, 0.0
    """
    key_norm = normalize_text(key)
    if not key_norm:
        return None, None, 0.0

    if key_norm in mapping:
        return mapping[key_norm], key_norm, 1.0

    candidates = list(mapping.keys())
    matches = difflib.get_close_matches(key_norm, candidates, n=1, cutoff=threshold)
    if not matches:
        return None, None, 0.0

    matched = matches[0]
    score = difflib.SequenceMatcher(None, key_norm, matched).ratio()
    return mapping[matched], matched, score


def merge_general_and_other_ml(topic: str) -> tuple[str, str | None]:
    """
    Return:
      final_topic, merged_from_topic

    merged_from_topic is not None only when General ML / Other ML is merged.
    """
    topic_norm = normalize_text(topic)
    if topic_norm in {
        normalize_text("Other topics in machine learning"),
        normalize_text("General machine learning"),
        normalize_text("other topics in machine learning (i.e., none of the above)"),
        normalize_text("general machine learning (i.e., none of the above)"),
    }:
        return MERGED_TOPIC_L2, topic

    return topic, None


def make_mapping_result(
    *,
    canonical_topic: str,
    canonical_topic_before_merge: str,
    needs_llm_refinement: bool,
    mapping_confidence: str,
    mapping_reason: str,
    matched_official_primary_area: str | None,
    mapping_score: float,
    llm_candidate_topics: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "canonical_topic": canonical_topic,
        "canonical_topic_before_merge": canonical_topic_before_merge,
        "needs_llm_refinement": needs_llm_refinement,
        "mapping_confidence": mapping_confidence,
        "mapping_reason": mapping_reason,
        "matched_official_primary_area": matched_official_primary_area,
        "mapping_score": round(mapping_score, 4),
        "llm_candidate_topics": llm_candidate_topics or [],
    }


def map_iclr_primary_area(
    official_primary_area: Any,
    fuzzy_threshold: float = 0.86,
) -> dict[str, Any]:
    mapped, matched_key, score = fuzzy_lookup(
        str(official_primary_area or ""),
        ICLR_PRIMARY_TO_CANONICAL,
        threshold=fuzzy_threshold,
    )

    if mapped is None:
        before_merge = "Other topics in machine learning"
        final_topic, _ = merge_general_and_other_ml(before_merge)
        return make_mapping_result(
            canonical_topic=final_topic,
            canonical_topic_before_merge=before_merge,
            needs_llm_refinement=True,
            mapping_confidence="fallback",
            mapping_reason=(
                "Unknown or empty ICLR official_primary_area; "
                "fallback to merged General/Other ML topic and mark for review."
            ),
            matched_official_primary_area=None,
            mapping_score=0.0,
            llm_candidate_topics=FINAL_CANONICAL_TOPICS,
        )

    before_merge = mapped
    final_topic, merged_from = merge_general_and_other_ml(before_merge)

    confidence = "direct" if score == 1.0 else "fuzzy_direct"
    needs_review = False
    reason = f"ICLR official_primary_area matched: {matched_key}"

    if merged_from is not None:
        reason += f"; merged from {merged_from!r} into {MERGED_TOPIC_L2!r}"

    return make_mapping_result(
        canonical_topic=final_topic,
        canonical_topic_before_merge=before_merge,
        needs_llm_refinement=needs_review,
        mapping_confidence=confidence,
        mapping_reason=reason,
        matched_official_primary_area=matched_key,
        mapping_score=score,
        llm_candidate_topics=[],
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object: {type(obj)}")

            rows.append(obj)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def count_with_percentage(counter: Counter[str], total: int) -> list[dict[str, Any]]:
    items = []
    for name, count in counter.most_common():
        items.append(
            {
                "name": name,
                "count": count,
                "percentage": round(count / total * 100, 4) if total else 0.0,
            }
        )
    return items


def build_summary(rows: list[dict[str, Any]], area_field: str) -> dict[str, Any]:
    total = len(rows)

    official_counter: Counter[str] = Counter()
    before_merge_counter: Counter[str] = Counter()
    final_counter: Counter[str] = Counter()
    confidence_counter: Counter[str] = Counter()

    need_review = 0
    fallback = 0

    for row in rows:
        official_counter[str(row.get(area_field, ""))] += 1
        before_merge_counter[str(row.get("canonical_topic_before_merge", ""))] += 1
        final_counter[str(row.get("canonical_topic", ""))] += 1
        confidence_counter[str(row.get("mapping_confidence", ""))] += 1

        if row.get("needs_llm_refinement") is True:
            need_review += 1
        if row.get("mapping_confidence") == "fallback":
            fallback += 1

    return {
        "total_records": total,
        "records_need_llm_refinement": need_review,
        "fallback_records": fallback,
        "by_official_primary_area": count_with_percentage(official_counter, total),
        "by_canonical_topic_before_merge": count_with_percentage(before_merge_counter, total),
        "by_canonical_topic": count_with_percentage(final_counter, total),
        "by_mapping_confidence": count_with_percentage(confidence_counter, total),
    }


def map_records(
    rows: list[dict[str, Any]],
    *,
    area_field: str,
    venue_field: str,
    default_venue: str,
    fuzzy_threshold: float,
) -> list[dict[str, Any]]:
    mapped_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        venue = normalize_venue(row.get(venue_field), default=default_venue)

        # This script is meant for ICLR rejected papers.
        # If venue is missing, default_venue="ICLR" makes it work.
        # If venue is explicitly non-ICLR, we still map by the ICLR area field but mark the reason.
        official_primary_area = row.get(area_field)

        mapping = map_iclr_primary_area(
            official_primary_area,
            fuzzy_threshold=fuzzy_threshold,
        )

        out = dict(row)

        # Reset output paper_id from 00001 in input order.
        # Keep the original paper_id for traceability.
        if "paper_id" in out:
            out["original_paper_id"] = out.get("paper_id")
        out["paper_id"] = f"{idx:05d}"

        out["venue_normalized"] = venue
        out.update(mapping)

        if venue != "ICLR":
            out["needs_llm_refinement"] = True
            out["mapping_reason"] += (
                f"; warning: venue_normalized={venue!r}, but this script uses ICLR mapping rules"
            )

        # Helpful stable row index for debugging bad lines or review rows.
        if "source_index" not in out:
            out["source_index"] = idx

        mapped_rows.append(out)

    return mapped_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach unified canonical topics to each paper in "
            "iclr2026_rejected_topics.jsonl."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL path, e.g. iclr2026_rejected_topics.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL path with mapped topic fields.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional summary JSON path.",
    )
    parser.add_argument(
        "--llm-todo-output",
        type=Path,
        default=None,
        help=(
            "Optional JSONL output containing records with "
            "needs_llm_refinement=True."
        ),
    )
    parser.add_argument(
        "--area-field",
        default="official_primary_area",
        help="Field name for ICLR official primary area.",
    )
    parser.add_argument(
        "--venue-field",
        default="venue",
        help="Field name for venue. Missing venue defaults to --default-venue.",
    )
    parser.add_argument(
        "--default-venue",
        default="ICLR",
        help="Default venue when venue field is missing or empty.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.86,
        help="Fuzzy match threshold, same default as the first script's ICLR mapping.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary JSON to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.input)
    mapped_rows = map_records(
        rows,
        area_field=args.area_field,
        venue_field=args.venue_field,
        default_venue=args.default_venue,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    write_jsonl(args.output, mapped_rows)

    summary = build_summary(mapped_rows, area_field=args.area_field)

    if args.summary_output is not None:
        write_json(args.summary_output, summary)

    if args.llm_todo_output is not None:
        todo_rows = [row for row in mapped_rows if row.get("needs_llm_refinement") is True]
        write_jsonl(args.llm_todo_output, todo_rows)

    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
