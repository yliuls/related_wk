#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from download_and_parse_openreview import (
    build_openreview_client,
    build_openreview_pdf_url,
    dedupe_path,
    download_pdf,
    download_pdf_via_openreview_api,
    parse_pdf_with_scipdf,
    sanitize_filename,
    truncate_scipdf_result,
)


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAPPING_PATH = DEFAULT_ROOT / "data" / "paper_id_mapping.json"
DEFAULT_CANDIDATE_PATH = DEFAULT_ROOT / "candidate_papers.json"
DEFAULT_PDF_ROOT = DEFAULT_ROOT / "pdfs"
DEFAULT_PARSED_JSON_ROOT = DEFAULT_ROOT / "parsed_json"
DEFAULT_PDF_REPORT_PATH = DEFAULT_PDF_ROOT / "download_status.json"
DEFAULT_JSON_REPORT_PATH = DEFAULT_PARSED_JSON_ROOT / "parse_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PDFs and parse them into JSON, grouped by paper ID ranges "
            "such as paper_0001_0010."
        )
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="Path to paper_id_mapping.json.",
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=DEFAULT_CANDIDATE_PATH,
        help="Path to candidate_papers.json.",
    )
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=DEFAULT_PDF_ROOT,
        help="Root directory for grouped PDF output folders.",
    )
    parser.add_argument(
        "--parsed-json-root",
        type=Path,
        default=DEFAULT_PARSED_JSON_ROOT,
        help="Root directory for grouped parsed JSON folders.",
    )
    parser.add_argument(
        "--pdf-report-path",
        type=Path,
        default=DEFAULT_PDF_REPORT_PATH,
        help="Path to write the PDF download status report.",
    )
    parser.add_argument(
        "--json-report-path",
        type=Path,
        default=DEFAULT_JSON_REPORT_PATH,
        help="Path to write the parsed JSON status report.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=10,
        help="Number of papers per folder group. Default: 10.",
    )
    parser.add_argument(
        "--start-paper-index",
        type=int,
        default=1,
        help="1-based inclusive paper index to start from. Default: 1.",
    )
    parser.add_argument(
        "--end-paper-index",
        type=int,
        default=None,
        help="1-based inclusive paper index to end at. Default: process all mapped papers.",
    )
    parser.add_argument(
        "--grobid-url",
        type=str,
        default="http://localhost:8070",
        help="GROBID service URL used by scipdf. Default: http://localhost:8070",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download or parse steps when the target file already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for PDF download. Default: 120.",
    )
    parser.add_argument(
        "--download-method",
        choices=["openreview-api", "direct", "auto"],
        default="auto",
        help="PDF download method. Default: auto.",
    )
    parser.add_argument(
        "--openreview-baseurl",
        type=str,
        default="https://api2.openreview.net",
        help="Base URL for the OpenReview official client. Default: https://api2.openreview.net",
    )
    parser.add_argument(
        "--openreview-username",
        type=str,
        default=os.environ.get("OPENREVIEW_USERNAME"),
        help="Optional OpenReview username/email for authenticated API download.",
    )
    parser.add_argument(
        "--openreview-password",
        type=str,
        default=os.environ.get("OPENREVIEW_PASSWORD"),
        help="Optional OpenReview password for authenticated API download.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_paper_numeric_id(paper_id: str) -> int:
    prefix = "paper_"
    if not paper_id.startswith(prefix):
        raise ValueError(f"Invalid paper_id: {paper_id}")
    suffix = paper_id[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError(f"Invalid paper_id: {paper_id}")
    return int(suffix)


def build_group_name(paper_numeric_id: int, group_size: int) -> str:
    group_start = ((paper_numeric_id - 1) // group_size) * group_size + 1
    group_end = group_start + group_size - 1
    return f"paper_{group_start:04d}_{group_end:04d}"


def build_title_to_paper_id(mapping_data: dict[str, Any]) -> dict[str, str]:
    title_to_paper_id = mapping_data.get("title_to_paper_id")
    if isinstance(title_to_paper_id, dict):
        return {
            str(title): str(paper_id)
            for title, paper_id in title_to_paper_id.items()
            if str(title).strip() and str(paper_id).strip()
        }

    papers = mapping_data.get("papers")
    if not isinstance(papers, list):
        raise ValueError("Mapping JSON must contain either title_to_paper_id or papers")

    result: dict[str, str] = {}
    for item in papers:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        paper_id = str(item.get("paper_id", "")).strip()
        if title and paper_id:
            result[title] = paper_id
    return result


def build_candidate_lookup(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in candidates:
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        if title in lookup:
            raise ValueError(f"Duplicate title in candidate JSON: {title}")
        lookup[title] = item
    return lookup


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def init_openreview_client(args: argparse.Namespace) -> Any | None:
    if args.download_method not in {"openreview-api", "auto"}:
        return None

    try:
        return build_openreview_client(
            baseurl=args.openreview_baseurl,
            username=args.openreview_username,
            password=args.openreview_password,
        )
    except Exception as exc:
        if args.download_method == "openreview-api":
            raise
        print(
            "Warning: could not initialize OpenReview client, "
            f"will fall back to direct download: {exc}",
            flush=True,
        )
        return None


def download_one_pdf(
    paper: dict[str, Any],
    pdf_path: Path,
    timeout: int,
    download_method: str,
    openreview_client: Any | None,
) -> tuple[bool, str]:
    title = str(paper.get("title", "")).strip() or pdf_path.stem
    openreview_url = paper.get("openreview_url")
    openreview_id = paper.get("openreview_id")
    if not openreview_url:
        return False, f"missing openreview_url for {title}"

    download_errors: list[str] = []
    downloaded = False

    if download_method in {"openreview-api", "auto"}:
        if not openreview_id:
            download_errors.append("missing openreview_id for official client download")
        elif openreview_client is None:
            download_errors.append("OpenReview client is not configured")
        else:
            try:
                download_pdf_via_openreview_api(openreview_client, str(openreview_id), pdf_path)
                downloaded = True
            except Exception as exc:
                download_errors.append(f"openreview-api failed: {exc}")
                if pdf_path.exists() and pdf_path.stat().st_size == 0:
                    pdf_path.unlink()

    if not downloaded and download_method in {"direct", "auto"}:
        try:
            pdf_url = build_openreview_pdf_url(str(openreview_url))
            download_pdf(pdf_url, pdf_path, timeout=timeout)
            downloaded = True
        except Exception as exc:
            download_errors.append(f"direct failed: {exc}")
            if pdf_path.exists() and pdf_path.stat().st_size == 0:
                pdf_path.unlink()

    if not downloaded:
        return False, "; ".join(download_errors)

    return True, "downloaded"


def parse_one_pdf(
    paper: dict[str, Any],
    pdf_path: Path,
    json_path: Path,
    grobid_url: str,
) -> tuple[bool, str]:
    try:
        parsed = parse_pdf_with_scipdf(pdf_path, grobid_url=grobid_url)
    except Exception as exc:
        return False, str(exc)

    if not isinstance(parsed, dict):
        return False, "SciPDF returned no parsed content"

    truncated = truncate_scipdf_result(parsed, paper)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(truncated, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return True, "parsed"


def main() -> int:
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if args.start_paper_index < 1:
        raise ValueError("--start-paper-index must be >= 1")

    mapping_data = load_json(args.mapping_json.expanduser().resolve())
    candidates = load_json(args.candidate_json.expanduser().resolve())
    if not isinstance(mapping_data, dict):
        raise ValueError("Mapping JSON must be an object")
    if not isinstance(candidates, list):
        raise ValueError("Candidate JSON must be a list")

    title_to_paper_id = build_title_to_paper_id(mapping_data)
    candidate_lookup = build_candidate_lookup(candidates)

    selected_records: list[dict[str, Any]] = []
    missing_candidates: list[dict[str, str]] = []
    for title, paper_id in sorted(title_to_paper_id.items(), key=lambda item: parse_paper_numeric_id(item[1])):
        candidate = candidate_lookup.get(title)
        if candidate is None:
            missing_candidates.append({"paper_id": paper_id, "title": title})
            continue
        paper_numeric_id = parse_paper_numeric_id(paper_id)
        if paper_numeric_id < args.start_paper_index:
            continue
        if args.end_paper_index is not None and paper_numeric_id > args.end_paper_index:
            continue
        selected_records.append(
            {
                "paper_id": paper_id,
                "paper_numeric_id": paper_numeric_id,
                "title": title,
                "paper": candidate,
                "group_name": build_group_name(paper_numeric_id, args.group_size),
            }
        )

    pdf_root = args.pdf_root.expanduser().resolve()
    parsed_json_root = args.parsed_json_root.expanduser().resolve()
    pdf_root.mkdir(parents=True, exist_ok=True)
    parsed_json_root.mkdir(parents=True, exist_ok=True)

    openreview_client = init_openreview_client(args)

    pdf_groups: dict[str, list[dict[str, Any]]] = {}
    json_groups: dict[str, list[dict[str, Any]]] = {}
    pdf_failed: list[dict[str, Any]] = []
    json_failed: list[dict[str, Any]] = []

    for record in selected_records:
        paper_id = record["paper_id"]
        title = record["title"]
        paper = record["paper"]
        group_name = record["group_name"]

        pdf_group_dir = pdf_root / group_name
        json_group_dir = parsed_json_root / group_name
        pdf_group_dir.mkdir(parents=True, exist_ok=True)
        json_group_dir.mkdir(parents=True, exist_ok=True)

        safe_name = sanitize_filename(title, fallback=paper_id)
        pdf_path = pdf_group_dir / f"{safe_name}.pdf"
        json_path = json_group_dir / f"{safe_name}.json"

        if pdf_path.exists() and not args.skip_existing:
            pdf_path = dedupe_path(pdf_path)
        if json_path.exists() and not args.skip_existing:
            json_path = json_group_dir / f"{pdf_path.stem}.json"

        pdf_status = "pending"
        pdf_message = ""

        if args.skip_existing and pdf_path.exists():
            pdf_status = "skipped_existing"
            pdf_message = "pdf already exists"
        else:
            ok, message = download_one_pdf(
                paper=paper,
                pdf_path=pdf_path,
                timeout=args.timeout,
                download_method=args.download_method,
                openreview_client=openreview_client,
            )
            pdf_status = "success" if ok else "failed"
            pdf_message = message

        pdf_entry = {
            "paper_id": paper_id,
            "title": title,
            "group_name": group_name,
            "pdf_path": str(pdf_path),
            "status": pdf_status,
            "message": pdf_message,
        }
        pdf_groups.setdefault(group_name, []).append(pdf_entry)
        if pdf_status == "failed":
            pdf_failed.append(pdf_entry)

        json_status = "pending"
        json_message = ""

        if pdf_status == "failed":
            json_status = "skipped_missing_pdf"
            json_message = "pdf download failed"
        elif args.skip_existing and json_path.exists():
            json_status = "skipped_existing"
            json_message = "json already exists"
        else:
            source_pdf_path = pdf_path
            ok, message = parse_one_pdf(
                paper=paper,
                pdf_path=source_pdf_path,
                json_path=json_path,
                grobid_url=args.grobid_url,
            )
            json_status = "success" if ok else "failed"
            json_message = message

        json_entry = {
            "paper_id": paper_id,
            "title": title,
            "group_name": group_name,
            "json_path": str(json_path),
            "source_pdf_path": str(pdf_path),
            "status": json_status,
            "message": json_message,
        }
        json_groups.setdefault(group_name, []).append(json_entry)
        if json_status == "failed":
            json_failed.append(json_entry)

        print(
            f"{paper_id} | {title} | pdf={pdf_status} | json={json_status}",
            flush=True,
        )

    pdf_report = {
        "mapping_json": str(args.mapping_json.expanduser().resolve()),
        "candidate_json": str(args.candidate_json.expanduser().resolve()),
        "pdf_root": str(pdf_root),
        "group_size": args.group_size,
        "start_paper_index": args.start_paper_index,
        "end_paper_index": args.end_paper_index,
        "total_selected": len(selected_records),
        "total_failed": len(pdf_failed),
        "missing_candidates": missing_candidates,
        "groups": pdf_groups,
        "failed_items": pdf_failed,
    }
    json_report = {
        "mapping_json": str(args.mapping_json.expanduser().resolve()),
        "candidate_json": str(args.candidate_json.expanduser().resolve()),
        "parsed_json_root": str(parsed_json_root),
        "group_size": args.group_size,
        "start_paper_index": args.start_paper_index,
        "end_paper_index": args.end_paper_index,
        "total_selected": len(selected_records),
        "total_failed": len(json_failed),
        "missing_candidates": missing_candidates,
        "groups": json_groups,
        "failed_items": json_failed,
    }

    write_report(args.pdf_report_path.expanduser().resolve(), pdf_report)
    write_report(args.json_report_path.expanduser().resolve(), json_report)

    print(
        "Finished grouped batch. "
        f"PDF failures={len(pdf_failed)}, JSON failures={len(json_failed)}",
        flush=True,
    )
    return 0 if not pdf_failed and not json_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
