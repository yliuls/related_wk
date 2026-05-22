#!/usr/bin/env python3
"""Sort L3 topic JSONL rows and add L2_topic from the topic map."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_L3_PATH = Path("iclr2026_rejected_l3_topics.jsonl")
DEFAULT_MAP_PATH = Path("iclr2026_rejected_topics_map.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sort iclr2026_rejected_l3_topics.jsonl by paper_id and add "
            "L2_topic from canonical_topic in iclr2026_rejected_topics_map.jsonl."
        )
    )
    parser.add_argument("--l3", type=Path, default=DEFAULT_L3_PATH, help="Input L3 JSONL file.")
    parser.add_argument(
        "--topic-map",
        type=Path,
        default=DEFAULT_MAP_PATH,
        help="JSONL file containing paper_id and canonical_topic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file. Omit when using --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the --l3 file atomically after writing a temporary file.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep rows with missing map entries and set L2_topic to null.",
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


def paper_sort_key(row: dict[str, Any]) -> tuple[int, int | str]:
    paper_id = str(row.get("paper_id", ""))
    if paper_id.isdigit():
        return (0, int(paper_id))
    return (1, paper_id)


def build_topic_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    topic_by_paper_id: dict[str, Any] = {}
    duplicate_ids: list[str] = []

    for row in rows:
        paper_id = row.get("paper_id")
        if paper_id is None:
            raise ValueError("topic map contains a row without paper_id")
        paper_id = str(paper_id)
        if paper_id in topic_by_paper_id:
            duplicate_ids.append(paper_id)
        topic_by_paper_id[paper_id] = row.get("canonical_topic")

    if duplicate_ids:
        examples = ", ".join(sorted(set(duplicate_ids))[:10])
        raise ValueError(f"topic map contains duplicate paper_id values: {examples}")

    return topic_by_paper_id


def add_l2_topic(row: dict[str, Any], l2_topic: Any) -> OrderedDict[str, Any]:
    output_row: OrderedDict[str, Any] = OrderedDict()
    inserted = False

    for key, value in row.items():
        if key == "L2_topic":
            continue
        if key == "L3topic_ids":
            output_row["L2_topic"] = l2_topic
            inserted = True
        output_row[key] = value

    if not inserted:
        output_row["L2_topic"] = l2_topic

    return output_row


def write_jsonl(path: Path, rows: list[OrderedDict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: Path, rows: list[OrderedDict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp_path = Path(f.name)
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(tmp_path, path)


def main() -> int:
    args = parse_args()
    if args.in_place and args.output is not None:
        raise SystemExit("Use either --in-place or --output, not both.")
    if not args.in_place and args.output is None:
        raise SystemExit("Please provide --output, or use --in-place.")

    l3_rows = load_jsonl(args.l3)
    topic_map_rows = load_jsonl(args.topic_map)
    topic_by_paper_id = build_topic_map(topic_map_rows)

    seen_l3_ids: set[str] = set()
    duplicate_l3_ids: list[str] = []
    missing_ids: list[str] = []
    output_rows: list[OrderedDict[str, Any]] = []

    for row in sorted(l3_rows, key=paper_sort_key):
        paper_id_value = row.get("paper_id")
        if paper_id_value is None:
            raise ValueError("L3 file contains a row without paper_id")

        paper_id = str(paper_id_value)
        if paper_id in seen_l3_ids:
            duplicate_l3_ids.append(paper_id)
        seen_l3_ids.add(paper_id)

        l2_topic = topic_by_paper_id.get(paper_id)
        if paper_id not in topic_by_paper_id:
            missing_ids.append(paper_id)

        output_rows.append(add_l2_topic(row, l2_topic))

    if duplicate_l3_ids:
        examples = ", ".join(sorted(set(duplicate_l3_ids))[:10])
        raise ValueError(f"L3 file contains duplicate paper_id values: {examples}")

    if missing_ids and not args.allow_missing:
        examples = ", ".join(missing_ids[:10])
        raise ValueError(
            "L3 rows missing from topic map: "
            f"{len(missing_ids)} total; examples: {examples}. "
            "Use --allow-missing to write null L2_topic for these rows."
        )

    if args.in_place:
        write_jsonl_atomic(args.l3, output_rows)
        output_path = args.l3
    else:
        write_jsonl(args.output, output_rows)
        output_path = args.output

    print(f"Wrote {len(output_rows)} rows to {output_path}")
    print("Rows are sorted by paper_id; L2_topic is inserted before L3topic_ids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
