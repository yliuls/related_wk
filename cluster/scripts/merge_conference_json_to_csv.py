#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topic_crawler.common import read_json_or_jsonl  # noqa: E402
from topic_crawler.schemas import STANDARD_FIELDS, normalize_record  # noqa: E402


def load_records(paths: list[str]) -> list[dict]:
    records = []
    for pattern in paths:
        matched = [Path(path) for path in sorted(glob.glob(pattern))] if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matched:
            if not path.exists() or path.name.endswith(".summary.json"):
                continue
            if path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            records.extend(normalize_record(record) for record in read_json_or_jsonl(path))
    return records


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_csv(records: list[dict], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: csv_value(record.get(field)) for field in STANDARD_FIELDS})
    tmp.replace(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge per-conference JSON files into one CSV.")
    parser.add_argument("--inputs", nargs="+", default=["data/outputs/*.json"], help="JSON/JSONL files or glob patterns.")
    parser.add_argument("--out", default="data/outputs/all_conference_topics.csv")
    args = parser.parse_args()

    records = load_records(args.inputs)
    write_csv(records, args.out)
    print(f"Wrote {len(records)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
