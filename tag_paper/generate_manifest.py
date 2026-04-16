#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manifest.jsonl for batch paper annotation."
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("/data3/yaofu/related_wk/tag_paper/paper_content"),
        help="Directory containing one Markdown file per paper.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data3/yaofu/related_wk/tag_paper/outputs"),
        help="Root directory for per-paper outputs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("/data3/yaofu/related_wk/tag_paper/data/manifest.jsonl"),
        help="Path to write manifest.jsonl.",
    )
    parser.add_argument(
        "--glob",
        default="*.md",
        help="File glob used to find paper files. Default: *.md",
    )
    return parser.parse_args()


def read_paper_title(paper_path: Path) -> str:
    with paper_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Title:"):
                return line.split("Title:", 1)[1].strip()
    return paper_path.stem


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve()

    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")

    papers = sorted(paper_dir.glob(args.glob))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        for idx, paper_path in enumerate(papers, start=1):
            paper_id = f"paper_{idx:04d}"
            paper_title = read_paper_title(paper_path)
            output_dir = output_root / paper_id
            record = {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "paper_path": str(paper_path),
                "output_dir": str(output_dir),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(papers)} items to {manifest_path}")


if __name__ == "__main__":
    main()
