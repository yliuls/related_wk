#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
import scipdf
from bs4 import NavigableString

try:
    import openreview
except ImportError:
    openreview = None


APPENDIX_HEADING_RE = re.compile(
    r"^\s*(appendix|appendices|supplementary|supplemental materials?)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PDFs from OpenReview URLs listed in a JSON file, then parse "
            "them with SciPDF Parser and save truncated JSON outputs."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to the candidate papers JSON file.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory to save downloaded PDFs.",
    )
    parser.add_argument(
        "--parsed-json-dir",
        type=Path,
        required=True,
        help="Directory to save parsed SciPDF JSON files.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based inclusive start index. Default: 1.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="1-based inclusive end index. Default: process to the end.",
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
        help="Skip papers whose PDF and parsed JSON already exist.",
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
        help=(
            "PDF download method. 'openreview-api' uses the official client, "
            "'direct' uses the OpenReview PDF URL, and 'auto' tries the official "
            "client first then falls back to direct download. Default: auto."
        ),
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
        help=(
            "Optional OpenReview username/email for authenticated API download. "
            "Defaults to env OPENREVIEW_USERNAME."
        ),
    )
    parser.add_argument(
        "--openreview-password",
        type=str,
        default=os.environ.get("OPENREVIEW_PASSWORD"),
        help=(
            "Optional OpenReview password for authenticated API download. "
            "Defaults to env OPENREVIEW_PASSWORD."
        ),
    )
    return parser.parse_args()


def load_papers(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def sanitize_filename(name: str, fallback: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        cleaned = fallback
    return cleaned[:180]


def dedupe_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def build_openreview_pdf_url(openreview_url: str) -> str:
    parsed = urlparse(openreview_url)
    query = parse_qs(parsed.query)
    paper_id = query.get("id", [None])[0]

    if parsed.path.endswith(".pdf"):
        return openreview_url
    if paper_id:
        return urlunparse(
            (
                parsed.scheme or "https",
                parsed.netloc or "openreview.net",
                "/pdf",
                "",
                urlencode({"id": paper_id}),
                "",
            )
        )
    raise ValueError(f"Could not extract OpenReview id from URL: {openreview_url}")


def truncate_sections_before_appendix(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for section in sections:
        heading = str(section.get("heading", "") or "")
        if APPENDIX_HEADING_RE.match(heading):
            break
        kept.append(section)
    return kept


def truncate_scipdf_result(parsed: dict[str, Any], paper_meta: dict[str, Any]) -> dict[str, Any]:
    result = dict(parsed)
    sections = result.get("sections")
    if isinstance(sections, list):
        result["sections"] = truncate_sections_before_appendix(sections)

    result["source_paper"] = {
        "title": paper_meta.get("title"),
        "openreview_url": paper_meta.get("openreview_url"),
        "openreview_id": paper_meta.get("openreview_id"),
        "venue": paper_meta.get("venue"),
        "year": paper_meta.get("year"),
    }
    return result


@contextlib.contextmanager
def bypass_local_proxy_for_url(url: str):
    hostname = (urlparse(url).hostname or "").lower()
    if hostname not in {"localhost", "127.0.0.1"}:
        yield
        return

    keys = [
        "NO_PROXY",
        "no_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    old_env = {key: os.environ.get(key) for key in keys}
    no_proxy_hosts = {"localhost", "127.0.0.1"}
    existing = old_env.get("NO_PROXY") or old_env.get("no_proxy") or ""
    if existing:
        no_proxy_hosts.update(part.strip() for part in existing.split(",") if part.strip())
    no_proxy_value = ",".join(sorted(no_proxy_hosts))

    os.environ["NO_PROXY"] = no_proxy_value
    os.environ["no_proxy"] = no_proxy_value
    for key in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _text_or_empty(node: Any) -> str:
    if node is None:
        return ""
    text = getattr(node, "text", "")
    return text.strip() if isinstance(text, str) else ""


def safe_parse_authors(article: Any) -> str:
    source_desc = article.find("sourcedesc")
    if source_desc is None:
        return ""
    authors = []
    for author in source_desc.find_all("persname"):
        firstname = _text_or_empty(author.find("forename", {"type": "first"}))
        middlename = _text_or_empty(author.find("forename", {"type": "middle"}))
        lastname = _text_or_empty(author.find("surname"))
        full_name = " ".join(part for part in [firstname, middlename, lastname] if part)
        if full_name:
            authors.append(full_name)
    return "; ".join(authors)


def safe_parse_date(article: Any) -> str:
    publication_stmt = article.find("publicationstmt")
    if publication_stmt is None:
        return ""
    date = publication_stmt.find("date")
    if date is None:
        return ""
    return date.attrs.get("when", "")


def safe_parse_abstract(article: Any) -> str:
    abstract = article.find("abstract")
    if abstract is None:
        return ""
    parts = []
    for child in abstract.children:
        if isinstance(child, NavigableString):
            continue
        text = _text_or_empty(child)
        if text:
            parts.append(text)
    return " ".join(parts)


def safe_parse_sections(article: Any) -> list[dict[str, Any]]:
    text_node = article.find("text")
    if text_node is None:
        return []
    divs = text_node.find_all("div", attrs={"xmlns": "http://www.tei-c.org/ns/1.0"})
    sections = []
    for div in divs:
        div_list = list(div.children)
        if len(div_list) == 0:
            heading = ""
            text = ""
        elif len(div_list) == 1:
            if isinstance(div_list[0], NavigableString):
                heading = str(div_list[0]).strip()
                text = ""
            else:
                heading = ""
                text = _text_or_empty(div_list[0])
        else:
            text_parts = []
            heading_candidate = div_list[0]
            if isinstance(heading_candidate, NavigableString):
                heading = str(heading_candidate).strip()
                p_all = div_list[1:]
            else:
                heading = ""
                p_all = div_list
            for p in p_all:
                try:
                    paragraph = _text_or_empty(p)
                except Exception:
                    paragraph = ""
                if paragraph:
                    text_parts.append(paragraph)
            text = "\n".join(text_parts)

        if heading or text:
            ref_dict = scipdf.find_references(div)
            sections.append(
                {
                    "heading": heading,
                    "text": text,
                    "publication_ref": ref_dict["publication_ref"],
                    "figure_ref": ref_dict["figure_ref"],
                    "table_ref": ref_dict["table_ref"],
                }
            )
    return sections


def safe_parse_references(article: Any) -> list[dict[str, Any]]:
    text_node = article.find("text")
    if text_node is None:
        return []
    references_div = text_node.find("div", attrs={"type": "references"})
    if references_div is None:
        return []
    reference_list = []
    for reference in references_div.find_all("biblstruct"):
        ref_id = reference.get("xml:id", "")
        title = reference.find("title", attrs={"level": "a"})
        if title is None:
            title = reference.find("title", attrs={"level": "m"})
        journal = reference.find("title", attrs={"level": "j"})
        if journal is None:
            journal = reference.find("publisher")
        year = reference.find("date")
        authors = []
        for author in reference.find_all("author"):
            firstname = _text_or_empty(author.find("forename", {"type": "first"}))
            middlename = _text_or_empty(author.find("forename", {"type": "middle"}))
            lastname = _text_or_empty(author.find("surname"))
            full_name = " ".join(part for part in [firstname, middlename, lastname] if part)
            if full_name:
                authors.append(full_name)
        reference_list.append(
            {
                "ref_id": ref_id,
                "title": _text_or_empty(title),
                "journal": _text_or_empty(journal),
                "year": year.attrs.get("when") if year is not None else "",
                "authors": "; ".join(authors),
            }
        )
    return reference_list


def safe_parse_figures(article: Any) -> list[dict[str, Any]]:
    figures_list = []
    for figure in article.find_all("figure"):
        figure_type = figure.attrs.get("type") or "figure"
        figure_id = figure.attrs.get("xml:id") or ""
        label = _text_or_empty(figure.find("label"))
        if figure_type == "table":
            caption = _text_or_empty(figure.find("figdesc"))
            table_node = figure.find("table")
            data = _text_or_empty(table_node)
        else:
            caption = _text_or_empty(figure)
            data = ""
        figures_list.append(
            {
                "figure_label": label,
                "figure_type": figure_type,
                "figure_id": figure_id,
                "figure_caption": caption,
                "figure_data": data,
            }
        )
    return figures_list


def safe_parse_formulas(article: Any) -> list[dict[str, Any]]:
    formulas_list = []
    for formula in article.find_all("formula"):
        formula_id = formula.attrs.get("xml:id") or ""
        formula_text = _text_or_empty(formula)
        formula_coordinates = formula.attrs.get("coords") or ""
        if formula_coordinates:
            try:
                coords = [float(x) for x in formula_coordinates.split(",")]
            except Exception:
                coords = []
            formulas_list.append(
                {
                    "formula_id": formula_id,
                    "formula_text": formula_text,
                    "formula_coordinates": coords,
                }
            )
    return formulas_list


def parse_pdf_with_scipdf(pdf_path: Path, grobid_url: str) -> dict[str, Any]:
    with bypass_local_proxy_for_url(grobid_url):
        article = scipdf.parse_pdf(
            str(pdf_path),
            fulltext=True,
            soup=True,
            return_coordinates=True,
            grobid_url=grobid_url,
        )

    if article is None:
        raise ValueError("GROBID returned no article content")

    title_node = article.find("title", attrs={"type": "main"})
    doi_node = article.find("idno", attrs={"type": "DOI"})

    return {
        "title": _text_or_empty(title_node),
        "authors": safe_parse_authors(article),
        "pub_date": safe_parse_date(article),
        "abstract": safe_parse_abstract(article),
        "sections": safe_parse_sections(article),
        "references": safe_parse_references(article),
        "figures": safe_parse_figures(article),
        "formulas": safe_parse_formulas(article),
        "doi": _text_or_empty(doi_node),
    }


def download_pdf(pdf_url: str, output_path: Path, timeout: int) -> None:
    with requests.get(
        pdf_url,
        stream=True,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            raise ValueError(
                f"Download did not look like a PDF. URL={pdf_url}, Content-Type={content_type}"
            )
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def build_openreview_client(
    baseurl: str,
    username: str | None,
    password: str | None,
):
    if openreview is None:
        raise RuntimeError(
            "openreview package is not installed. Install openreview-py first."
        )
    kwargs: dict[str, Any] = {"baseurl": baseurl}
    if username and password:
        kwargs["username"] = username
        kwargs["password"] = password
    return openreview.api.OpenReviewClient(**kwargs)


def download_pdf_via_openreview_api(
    client: Any,
    note_id: str,
    output_path: Path,
) -> None:
    pdf_bytes = client.get_pdf(note_id)
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise ValueError(f"OpenReview client returned unexpected payload type: {type(pdf_bytes).__name__}")
    with output_path.open("wb") as f:
        f.write(pdf_bytes)


def process_paper(
    paper: dict[str, Any],
    index: int,
    pdf_dir: Path,
    parsed_json_dir: Path,
    grobid_url: str,
    skip_existing: bool,
    timeout: int,
    download_method: str,
    openreview_client: Any | None,
) -> tuple[bool, str]:
    title = str(paper.get("title") or f"paper_{index}")
    openreview_url = paper.get("openreview_url")
    openreview_id = paper.get("openreview_id")
    if not openreview_url:
        return False, f"[{index}] missing openreview_url: {title}"

    safe_name = sanitize_filename(title, fallback=f"paper_{index}")
    pdf_path = pdf_dir / f"{safe_name}.pdf"
    json_path = parsed_json_dir / f"{safe_name}.json"

    if skip_existing and pdf_path.exists() and json_path.exists():
        return True, f"[{index}] skipped existing: {title}"

    if pdf_path.exists() and not json_path.exists():
        final_pdf_path = pdf_path
    elif not pdf_path.exists():
        final_pdf_path = pdf_path
    else:
        final_pdf_path = dedupe_path(pdf_path)
        json_path = parsed_json_dir / f"{final_pdf_path.stem}.json"

    download_errors = []
    downloaded = False

    if download_method in {"openreview-api", "auto"}:
        if not openreview_id:
            download_errors.append("missing openreview_id for official client download")
        elif openreview_client is None:
            download_errors.append("OpenReview client is not configured")
        else:
            try:
                download_pdf_via_openreview_api(openreview_client, str(openreview_id), final_pdf_path)
                downloaded = True
            except Exception as exc:
                download_errors.append(f"openreview-api failed: {exc}")
                if final_pdf_path.exists() and final_pdf_path.stat().st_size == 0:
                    final_pdf_path.unlink()

    if not downloaded and download_method in {"direct", "auto"}:
        try:
            pdf_url = build_openreview_pdf_url(str(openreview_url))
            download_pdf(pdf_url, final_pdf_path, timeout=timeout)
            downloaded = True
        except Exception as exc:
            download_errors.append(f"direct failed: {exc}")
            if final_pdf_path.exists() and final_pdf_path.stat().st_size == 0:
                final_pdf_path.unlink()

    if not downloaded:
        return False, f"[{index}] download failed for {title}: {'; '.join(download_errors)}"

    try:
        parsed = parse_pdf_with_scipdf(final_pdf_path, grobid_url=grobid_url)
    except Exception as exc:
        return False, f"[{index}] SciPDF parse failed for {title}: {exc}"

    if not isinstance(parsed, dict):
        return False, f"[{index}] SciPDF returned no parsed content for {title}"

    truncated = truncate_scipdf_result(parsed, paper)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(truncated, f, ensure_ascii=False, indent=2)

    return True, f"[{index}] done: {title}"


def main() -> int:
    args = parse_args()
    papers = load_papers(args.input_json)

    if args.start < 1:
        raise ValueError("--start must be >= 1")

    end = args.end if args.end is not None else len(papers)
    if end < args.start:
        raise ValueError("--end must be >= --start")

    selected = papers[args.start - 1 : end]
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.parsed_json_dir.mkdir(parents=True, exist_ok=True)

    openreview_client = None
    if args.download_method in {"openreview-api", "auto"}:
        try:
            openreview_client = build_openreview_client(
                baseurl=args.openreview_baseurl,
                username=args.openreview_username,
                password=args.openreview_password,
            )
        except Exception as exc:
            if args.download_method == "openreview-api":
                raise
            print(
                f"Warning: could not initialize OpenReview client, will fall back to direct download: {exc}",
                flush=True,
            )

    total = len(selected)
    success_count = 0
    failure_count = 0

    for offset, paper in enumerate(selected, start=args.start):
        try:
            ok, message = process_paper(
                paper=paper,
                index=offset,
                pdf_dir=args.pdf_dir,
                parsed_json_dir=args.parsed_json_dir,
                grobid_url=args.grobid_url,
                skip_existing=args.skip_existing,
                timeout=args.timeout,
                download_method=args.download_method,
                openreview_client=openreview_client,
            )
        except Exception as exc:
            ok = False
            title = paper.get("title", f"paper_{offset}")
            message = f"[{offset}] failed: {title}: {exc}"

        print(message, flush=True)
        if ok:
            success_count += 1
        else:
            failure_count += 1

    print(
        f"Finished {total} papers. Success={success_count}, Failure={failure_count}",
        flush=True,
    )
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
