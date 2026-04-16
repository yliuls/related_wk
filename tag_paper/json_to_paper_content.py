#!/usr/bin/env python3
"""
Convert one paper JSON file or a directory of paper JSON files into markdown.

Each output markdown contains:
- Title
- Abstract
- All sections
- Reference list

Usage:
    python json_to_paper_content.py \
        --input "/path/to/paper.json" \
        --output_dir "/path/to/output_dir"

    python json_to_paper_content.py \
        --input "/path/to/json_dir" \
        --output_dir "/path/to/output_dir"

Output:
    /path/to/output_dir/<same_json_basename>.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_paper_title_abstract(article: Dict[str, Any]) -> str:
    title = str(article.get("title", "")).strip()
    abstract = str(article.get("abstract", "")).strip()
    paper_content = (
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
    )
    return paper_content


def read_paper_content(article: Dict[str, Any]) -> str:
    paper_content = read_paper_title_abstract(article)

    sections = article.get("sections", []) or []
    for section in sections:
        heading = str(section.get("heading", "")).strip()
        text = str(section.get("text", "")).strip()
        publication_ref = section.get("publication_ref", [])

        if not isinstance(publication_ref, list):
            publication_ref = [publication_ref] if publication_ref else []

        paper_content += (
            f"\nSection: {heading}\n"
            f"{text}\n"
            f"this section cite: {publication_ref}\n"
        )

    return paper_content


def read_paper_content_with_ref(article: Dict[str, Any]) -> str:
    paper_content = read_paper_content(article)
    paper_content += "\nSection: References\n"

    references = article.get("references", []) or []
    for refer in references:
        ref_id = str(refer.get("ref_id", "")).strip()
        title = str(refer.get("title", "")).strip()
        year = refer.get("year", "")
        year_str = "" if year is None else str(year).strip()
        paper_content += f"Ref_id:{ref_id} Title: {title} Year: ({year_str})\n"

    return paper_content


def ensure_single_article(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        raise ValueError(
            "The input JSON is a list with multiple items. "
            "This script expects one paper per JSON file."
        )

    raise ValueError("Unsupported JSON format. Expected a dict or a single-item list.")


def save_markdown(content: str, input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def convert_one_json(input_path: Path, output_dir: Path) -> tuple[str, Path]:
    output_path = output_dir / f"{input_path.stem}.md"
    if output_path.exists():
        return "skipped", output_path

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    article = ensure_single_article(data)
    paper_content = read_paper_content_with_ref(article)
    saved_path = save_markdown(paper_content, input_path, output_dir)
    return "saved", saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a paper JSON file to a markdown paper_content file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an input JSON file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the output markdown file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Input file must be a .json file: {input_path}")
        status, output_path = convert_one_json(input_path, output_dir)
        if status == "saved":
            print(f"Saved to: {output_path}")
        else:
            print(f"Skipped existing: {output_path}")
        return

    if not input_path.is_dir():
        raise ValueError(f"Input path must be a JSON file or a directory: {input_path}")

    json_files = sorted(p for p in input_path.glob("*.json") if p.is_file())
    saved_count = 0
    skipped_count = 0

    for json_file in json_files:
        status, output_path = convert_one_json(json_file, output_dir)
        if status == "saved":
            saved_count += 1
            print(f"Saved to: {output_path}")
        else:
            skipped_count += 1
            print(f"Skipped existing: {output_path}")

    print(
        f"Finished {len(json_files)} files. Saved={saved_count}, Skipped={skipped_count}"
    )


if __name__ == "__main__":
    main()
