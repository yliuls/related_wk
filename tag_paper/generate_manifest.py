#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_PAPER_DIR = DEFAULT_ROOT / "paper_content"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "outputs"
DEFAULT_MANIFEST_PATH = DEFAULT_ROOT / "data" / "manifest.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manifest.jsonl for batch paper annotation."
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=DEFAULT_PAPER_DIR,
        help="Directory containing one Markdown file per paper.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for per-paper outputs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
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


def to_manifest_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve()
    root_dir = manifest_path.parent.parent.resolve()

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
                "paper_path": to_manifest_path(paper_path, root_dir),
                "output_dir": to_manifest_path(output_dir, root_dir),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(papers)} items to {manifest_path}")


if __name__ == "__main__":
    main()
