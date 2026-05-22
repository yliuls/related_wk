#!/usr/bin/env python3
"""Build an L2/L3 topic summary JSONL from per-paper ICLR rejected L3 topics."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("iclr2026_rejected_l3_topics.jsonl")
DEFAULT_OUTPUT = Path("iclr2026_rejected_l3_topics_all.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-paper ICLR rejected L3 topic assignments into an "
            "ml_l3_topics_all.jsonl-like L2/L3 summary."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Per-paper L3 JSONL input.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Grouped L2/L3 JSONL output.",
    )
    parser.add_argument(
        "--sort-l2",
        choices=("paper_id", "name", "total_papers"),
        default="paper_id",
        help="Sort L2 rows by first paper_id, L2 name, or descending total_papers.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(obj)
    return rows


def paper_sort_key(paper_id: str) -> tuple[int, int | str]:
    if paper_id.isdigit():
        return (0, int(paper_id))
    return (1, paper_id)


def topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    if len(topic_id) > 1 and topic_id[0].upper() == "T" and topic_id[1:].isdigit():
        return (0, int(topic_id[1:]))
    return (1, topic_id)


def normalize_l3_entries(row: dict[str, Any]) -> list[dict[str, str]]:
    details = row.get("L3topic_detail")
    if isinstance(details, list) and details:
        entries: list[dict[str, str]] = []
        for item in details:
            if not isinstance(item, dict):
                continue
            topic_id = item.get("id")
            name_en = item.get("name_en") or item.get("name") or item.get("name_zh")
            if topic_id is None:
                continue
            entries.append({"id": str(topic_id), "name_en": "" if name_en is None else str(name_en)})
        return entries

    topic_ids = row.get("L3topic_ids") or []
    topic_names = row.get("L3topic") or []
    entries = []
    for index, topic_id in enumerate(topic_ids):
        if topic_id is None:
            continue
        name_en = topic_names[index] if index < len(topic_names) else ""
        entries.append({"id": str(topic_id), "name_en": "" if name_en is None else str(name_en)})
    return entries


def build_summary(rows: list[dict[str, Any]], sort_l2: str) -> list[OrderedDict[str, Any]]:
    papers_by_l2: dict[str, set[str]] = defaultdict(set)
    l3_by_l2: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    first_l2_paper_id: dict[str, str] = {}

    for row in rows:
        paper_id_value = row.get("paper_id")
        if paper_id_value is None:
            raise ValueError("input contains a row without paper_id")
        paper_id = str(paper_id_value)

        l2_topic_value = row.get("L2_topic")
        l2_topic = "Unmapped" if l2_topic_value in (None, "") else str(l2_topic_value)
        papers_by_l2[l2_topic].add(paper_id)
        if l2_topic not in first_l2_paper_id or paper_sort_key(paper_id) < paper_sort_key(first_l2_paper_id[l2_topic]):
            first_l2_paper_id[l2_topic] = paper_id

        seen_l3_in_paper: set[str] = set()
        for entry in normalize_l3_entries(row):
            topic_id = entry["id"]
            if topic_id in seen_l3_in_paper:
                continue
            seen_l3_in_paper.add(topic_id)

            topic = l3_by_l2[l2_topic].setdefault(
                topic_id,
                {
                    "id": topic_id,
                    "name_en": entry["name_en"],
                    "paper_ids": set(),
                },
            )
            if not topic["name_en"] and entry["name_en"]:
                topic["name_en"] = entry["name_en"]
            topic["paper_ids"].add(paper_id)

    def l2_sort_key(l2_topic: str) -> tuple[Any, ...]:
        if sort_l2 == "name":
            return (l2_topic,)
        if sort_l2 == "total_papers":
            return (-len(papers_by_l2[l2_topic]), l2_topic)
        return (*paper_sort_key(first_l2_paper_id[l2_topic]), l2_topic)

    output_rows: list[OrderedDict[str, Any]] = []
    for l2_topic in sorted(papers_by_l2, key=l2_sort_key):
        l3_topics = []
        for topic_id in sorted(l3_by_l2[l2_topic], key=topic_sort_key):
            topic = l3_by_l2[l2_topic][topic_id]
            paper_ids = sorted(topic["paper_ids"], key=paper_sort_key)
            name_en = topic["name_en"]
            l3_topics.append(
                OrderedDict(
                    [
                        ("id", topic_id),
                        ("name", name_en),
                        ("name_en", name_en),
                        ("description", ""),
                        ("paper_count", len(paper_ids)),
                        ("paper_ids", paper_ids),
                        ("description_en", ""),
                    ]
                )
            )

        output_rows.append(
            OrderedDict(
                [
                    ("topic_L2", l2_topic),
                    ("total_papers", len(papers_by_l2[l2_topic])),
                    ("num_topics", len(l3_topics)),
                    ("topics_L3", l3_topics),
                ]
            )
        )

    return output_rows


def write_jsonl(path: Path, rows: list[OrderedDict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    summary_rows = build_summary(rows, args.sort_l2)
    write_jsonl(args.output, summary_rows)

    total_papers = sum(row["total_papers"] for row in summary_rows)
    total_l3_topics = sum(row["num_topics"] for row in summary_rows)
    print(f"Wrote {len(summary_rows)} L2 rows to {args.output}")
    print(f"Counted {total_papers} papers across L2 topics and {total_l3_topics} L3 topic groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
