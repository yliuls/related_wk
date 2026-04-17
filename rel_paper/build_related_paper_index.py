#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan subdirectories for related_work.json files and build "
            "paper-to-related-paper mappings in both directions."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing paper subdirectories, each with related_work.json.",
    )
    parser.add_argument(
        "--paper-to-related-output",
        type=Path,
        default=Path("paper_to_related_papers.json"),
        help="Output JSON mapping paper_idx to its related papers.",
    )
    parser.add_argument(
        "--related-to-paper-output",
        type=Path,
        default=Path("related_paper_to_citing_papers.json"),
        help="Output JSON mapping related paper title to citing paper_idx list.",
    )
    return parser.parse_args()


def load_related_work(related_work_path: Path) -> dict:
    with related_work_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_title(title: str) -> str:
    return " ".join(title.split())


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    paper_to_related: dict[str, dict] = {}
    related_to_papers: defaultdict[str, set[str]] = defaultdict(set)
    processed_dirs = 0
    missing_files = []

    for paper_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        processed_dirs += 1
        paper_idx = paper_dir.name
        related_work_path = paper_dir / "related_work.json"

        if not related_work_path.exists():
            missing_files.append(str(related_work_path))
            continue

        payload = load_related_work(related_work_path)
        target_paper_title = payload.get("target_paper_title", "")
        related_work_items = payload.get("related_work_items", [])

        related_titles = []
        seen_titles = set()
        for item in related_work_items:
            raw_title = item.get("paper_title", "")
            title = normalize_title(raw_title)
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            related_titles.append(title)
            related_to_papers[title].add(paper_idx)

        paper_to_related[paper_idx] = {
            "target_paper_title": target_paper_title,
            "related_papers": related_titles,
            "related_paper_count": len(related_titles),
        }

    paper_output = args.paper_to_related_output.expanduser()
    related_output = args.related_to_paper_output.expanduser()
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    related_output.parent.mkdir(parents=True, exist_ok=True)

    related_to_papers_json = {
        title: {
            "citing_papers": sorted(paper_idxs),
            "citation_count": len(paper_idxs),
        }
        for title, paper_idxs in sorted(related_to_papers.items())
    }

    with paper_output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "processed_subdirectories": processed_dirs,
                "papers_with_related_work": len(paper_to_related),
                "missing_related_work_files": missing_files,
                "paper_to_related_papers": paper_to_related,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    with related_output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "unique_related_papers": len(related_to_papers_json),
                "related_paper_to_citing_papers": related_to_papers_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print(f"Processed {processed_dirs} subdirectories under {input_dir}")
    print(f"Wrote paper-to-related mapping to {paper_output.resolve()}")
    print(f"Wrote related-to-paper mapping to {related_output.resolve()}")
    if missing_files:
        print(f"Skipped {len(missing_files)} directories without related_work.json")


if __name__ == "__main__":
    main()
