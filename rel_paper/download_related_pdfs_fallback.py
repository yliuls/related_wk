#!/usr/bin/env python3
"""
Batch-download PDFs for titles listed in related_paper_to_citing_papers.json.

Features
--------
- Reads paper titles from the JSON mapping.
- Tries multiple resolvers in configurable order:
    semantic_scholar -> arxiv -> openreview
- Retries on HTTP 429 / 5xx with exponential backoff.
- Supports optional OpenReview metadata (title -> openreview_id/url).
- Can also try OpenReview title search through openreview-py when installed.
- Saves a JSONL log for resumability / auditing.

This script only attempts open/publicly reachable PDFs. It does not bypass login walls,
paywalls, CAPTCHAs, or access restrictions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import requests

try:
    import openreview
except ImportError:
    openreview = None

USER_AGENT = "related-pdf-downloader/0.2"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
DEFAULT_TIMEOUT = 30
PDF_MAGIC = b"%PDF"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_filename(name: str, max_len: int = 180) -> str:
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[\\/:*?\"<>|\n\r\t]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "untitled"


def title_similarity(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


@dataclass
class DownloadResult:
    title: str
    citing_papers: List[str]
    status: str
    source: Optional[str] = None
    matched_title: Optional[str] = None
    score: Optional[float] = None
    download_url: Optional[str] = None
    saved_path: Optional[str] = None
    note: Optional[str] = None


class PdfDownloader:
    def __init__(
        self,
        outdir: Path,
        min_match_score: float = 0.88,
        sleep_s: float = 1.0,
        overwrite: bool = False,
        s2_api_key: Optional[str] = None,
        fallback_order: Optional[List[str]] = None,
        openreview_meta: Optional[Dict[str, Dict[str, str]]] = None,
        openreview_baseurl: str = "https://api2.openreview.net",
        openreview_username: Optional[str] = None,
        openreview_password: Optional[str] = None,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_cap: float = 60.0,
    ):
        self.outdir = outdir
        self.min_match_score = min_match_score
        self.sleep_s = sleep_s
        self.overwrite = overwrite
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if s2_api_key:
            self.session.headers.update({"x-api-key": s2_api_key})
        self.fallback_order = fallback_order or ["semantic_scholar", "arxiv", "openreview"]
        self.openreview_meta = openreview_meta or {}
        self.openreview_baseurl = openreview_baseurl
        self.openreview_username = openreview_username
        self.openreview_password = openreview_password
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._openreview_client = None

    def _sleep_for_retry(self, resp: Optional[requests.Response], attempt: int) -> None:
        retry_after = None
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                wait_s = float(retry_after)
            except ValueError:
                wait_s = min(self.backoff_cap, self.backoff_base ** attempt)
        else:
            wait_s = min(self.backoff_cap, self.backoff_base ** attempt)
        time.sleep(wait_s)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            resp: Optional[requests.Response] = None
            try:
                resp = self.session.request(method, url, timeout=DEFAULT_TIMEOUT, **kwargs)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    if attempt == self.max_retries - 1:
                        resp.raise_for_status()
                    self._sleep_for_retry(resp, attempt)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_err = exc
                if attempt == self.max_retries - 1:
                    raise
                self._sleep_for_retry(resp, attempt)
        assert last_err is not None
        raise last_err

    def request_json(self, url: str, **kwargs) -> dict:
        resp = self.request("GET", url, **kwargs)
        return resp.json()

    def semantic_scholar_search(self, title: str, limit: int = 10) -> List[dict]:
        params = {
            "query": title,
            "limit": limit,
            "fields": "title,year,externalIds,openAccessPdf,url,publicationDate",
        }
        data = self.request_json(S2_SEARCH_URL, params=params)
        return data.get("data", []) or []

    def best_semantic_scholar_match(self, title: str) -> Optional[Tuple[dict, float]]:
        candidates = self.semantic_scholar_search(title, limit=10)
        if not candidates:
            return None

        scored: List[Tuple[dict, float]] = []
        for cand in candidates:
            cand_title = cand.get("title") or ""
            score = title_similarity(title, cand_title)
            scored.append((cand, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0]

    def find_pdf_via_semantic_scholar(
        self, title: str
    ) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str], Optional[str]]:
        best = self.best_semantic_scholar_match(title)
        if not best:
            return None, None, None, None, "semantic_scholar_no_match"

        cand, score = best
        matched_title = cand.get("title") or ""
        if score < self.min_match_score:
            return None, None, score, matched_title, f"semantic_scholar_low_score<{self.min_match_score}"

        oa = cand.get("openAccessPdf") or {}
        pdf_url = oa.get("url")
        if pdf_url:
            return pdf_url, "semantic_scholar_openAccessPdf", score, matched_title, None

        ext = cand.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv") or ext.get("ARXIV")
        if arxiv_id:
            arxiv_id = arxiv_id.replace("arXiv:", "").strip()
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf", "semantic_scholar_arxiv", score, matched_title, None

        doi = ext.get("DOI")
        if doi:
            pdf_url = self.try_find_pdf_from_doi(doi)
            if pdf_url:
                return pdf_url, "doi_landing_page", score, matched_title, None

        return None, None, score, matched_title, "semantic_scholar_no_open_pdf"

    def arxiv_title_search(self, title: str, max_results: int = 5) -> List[dict]:
        query = f'ti:"{title}"'
        params = {"search_query": query, "start": 0, "max_results": max_results}
        resp = self.request("GET", ARXIV_API_URL, params=params)
        root = ET.fromstring(resp.text)

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        items: List[dict] = []
        for entry in root.findall("atom:entry", ns):
            item_title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            pdf_link = None
            for link in entry.findall("atom:link", ns):
                title_attr = link.attrib.get("title")
                href = link.attrib.get("href")
                if title_attr == "pdf" and href:
                    pdf_link = href
                    break
            items.append({"title": item_title, "id": entry_id, "pdf_url": pdf_link})
        return items

    def find_pdf_via_arxiv_search(
        self, title: str
    ) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str], Optional[str]]:
        try:
            items = self.arxiv_title_search(title, max_results=5)
        except Exception as exc:
            return None, None, None, None, f"arxiv_search_error:{exc}"

        if not items:
            return None, None, None, None, "arxiv_no_match"

        scored = []
        for item in items:
            score = title_similarity(title, item.get("title") or "")
            scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        best, score = scored[0]

        if score < self.min_match_score:
            return None, None, score, best.get("title"), f"arxiv_low_score<{self.min_match_score}"

        pdf_url = best.get("pdf_url")
        if pdf_url and not pdf_url.endswith(".pdf"):
            pdf_url = pdf_url + ".pdf"
        return pdf_url, "arxiv_search", score, best.get("title"), None

    def try_find_pdf_from_doi(self, doi: str) -> Optional[str]:
        doi = doi.strip()
        if not doi:
            return None

        landing_url = f"https://doi.org/{quote(doi, safe='/')}"
        try:
            resp = self.request("GET", landing_url, allow_redirects=True)
        except Exception:
            return None

        ctype = (resp.headers.get("content-type") or "").lower()
        if "application/pdf" in ctype:
            return resp.url

        html = resp.text or ""
        meta_patterns = [
            r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+property=["\']og:pdf["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']dc\.identifier["\'][^>]+content=["\']([^"\']+\.pdf)["\']',
        ]
        for pat in meta_patterns:
            m = re.search(pat, html, flags=re.I)
            if m:
                return m.group(1)

        href_patterns = [
            r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
            r'href=["\']([^"\']+/pdf(?:/|\?|$)[^"\']*)["\']',
        ]
        for pat in href_patterns:
            m = re.search(pat, html, flags=re.I)
            if m:
                return m.group(1)
        return None

    def _get_openreview_client(self):
        if self._openreview_client is not None:
            return self._openreview_client
        if openreview is None:
            return None
        kwargs: Dict[str, Any] = {"baseurl": self.openreview_baseurl}
        if self.openreview_username and self.openreview_password:
            kwargs["username"] = self.openreview_username
            kwargs["password"] = self.openreview_password
        try:
            self._openreview_client = openreview.api.OpenReviewClient(**kwargs)
        except Exception:
            self._openreview_client = None
        return self._openreview_client

    def _extract_openreview_id(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        value = value.strip()
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            qs = parse_qs(parsed.query)
            if "id" in qs and qs["id"]:
                return qs["id"][0]
            if parsed.path.endswith(".pdf"):
                base = Path(parsed.path).name
                if base and base.lower().endswith(".pdf"):
                    return base[:-4] or None
        return value if "/" not in value and "?" not in value else None

    def _build_openreview_pdf_url(self, note_id: str) -> str:
        return f"https://openreview.net/pdf?id={quote(note_id, safe='')}"

    def find_pdf_via_openreview_metadata(
        self, title: str
    ) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str], Optional[str]]:
        hit = self.openreview_meta.get(normalize_text(title))
        if not hit:
            return None, None, None, None, "openreview_metadata_no_match"

        matched_title = hit.get("title") or title
        score = title_similarity(title, matched_title)
        if score < self.min_match_score:
            return None, None, score, matched_title, f"openreview_metadata_low_score<{self.min_match_score}"

        note_id = self._extract_openreview_id(hit.get("openreview_id") or hit.get("openreview_url"))
        if note_id:
            return self._build_openreview_pdf_url(note_id), "openreview_metadata", score, matched_title, None

        pdf_url = hit.get("pdf_url")
        if pdf_url:
            return pdf_url, "openreview_metadata", score, matched_title, None

        return None, None, score, matched_title, "openreview_metadata_missing_id"

    def find_pdf_via_openreview_api(
        self, title: str
    ) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str], Optional[str]]:
        client = self._get_openreview_client()
        if client is None:
            return None, None, None, None, "openreview_client_unavailable"

        notes: List[Any] = []
        last_exc: Optional[Exception] = None
        methods = ["get_all_notes", "get_notes"]
        for method_name in methods:
            fn = getattr(client, method_name, None)
            if fn is None:
                continue
            try:
                notes = fn(content={"title": title}, limit=10) or []
                if notes:
                    break
            except TypeError:
                try:
                    notes = fn(content={"title": title}) or []
                    if notes:
                        break
                except Exception as exc:
                    last_exc = exc
            except Exception as exc:
                last_exc = exc

        if not notes:
            if last_exc is not None:
                return None, None, None, None, f"openreview_search_error:{last_exc}"
            return None, None, None, None, "openreview_no_match"

        scored = []
        for note in notes:
            cand_title = ""
            cand_id = None
            try:
                if hasattr(note, "content") and isinstance(note.content, dict):
                    cand_title = note.content.get("title") or ""
                elif isinstance(note, dict):
                    cand_title = (note.get("content") or {}).get("title") or note.get("title") or ""
                cand_id = getattr(note, "id", None) or (note.get("id") if isinstance(note, dict) else None)
            except Exception:
                pass
            if not cand_title or not cand_id:
                continue
            score = title_similarity(title, cand_title)
            scored.append((cand_title, cand_id, score))

        if not scored:
            return None, None, None, None, "openreview_no_usable_notes"

        scored.sort(key=lambda x: x[2], reverse=True)
        matched_title, note_id, score = scored[0]
        if score < self.min_match_score:
            return None, None, score, matched_title, f"openreview_low_score<{self.min_match_score}"
        return self._build_openreview_pdf_url(str(note_id)), "openreview_api_search", score, matched_title, None

    def find_pdf_via_openreview(
        self, title: str
    ) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str], Optional[str]]:
        url, source, score, matched_title, note = self.find_pdf_via_openreview_metadata(title)
        if url:
            return url, source, score, matched_title, note
        url2, source2, score2, matched_title2, note2 = self.find_pdf_via_openreview_api(title)
        if url2:
            return url2, source2, score2, matched_title2, note2
        notes = ";".join(x for x in [note, note2] if x)
        return None, None, score2 or score, matched_title2 or matched_title, notes or "openreview_no_match"

    def is_probably_pdf_response(self, resp: requests.Response, initial_bytes: bytes) -> bool:
        ctype = (resp.headers.get("content-type") or "").lower()
        if "application/pdf" in ctype:
            return True
        return initial_bytes.startswith(PDF_MAGIC)

    def download_pdf(self, url: str, dest: Path) -> None:
        if dest.exists() and not self.overwrite:
            return
        if dest.exists() and self.overwrite:
            dest.unlink()

        with self.request("GET", url, stream=True, allow_redirects=True) as resp:
            first_chunk = b""
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    if not first_chunk:
                        first_chunk = chunk[:16]
                        if not self.is_probably_pdf_response(resp, first_chunk):
                            raise ValueError(
                                f"response_is_not_pdf: content-type={resp.headers.get('content-type')}"
                            )
                    f.write(chunk)

        if dest.stat().st_size == 0:
            raise ValueError("downloaded_empty_file")

    def resolve_one(
        self, title: str
    ) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str], Optional[str]]:
        notes: List[str] = []
        best_score: Optional[float] = None
        best_title: Optional[str] = None

        for resolver in self.fallback_order:
            try:
                if resolver == "semantic_scholar":
                    url, source, score, matched_title, note = self.find_pdf_via_semantic_scholar(title)
                elif resolver == "arxiv":
                    url, source, score, matched_title, note = self.find_pdf_via_arxiv_search(title)
                elif resolver == "openreview":
                    url, source, score, matched_title, note = self.find_pdf_via_openreview(title)
                else:
                    notes.append(f"unknown_resolver:{resolver}")
                    continue
            except Exception as exc:
                notes.append(f"{resolver}_error:{exc}")
                continue

            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_title = matched_title
            if note:
                notes.append(note)
            if url:
                return url, source, score, matched_title, ";".join(notes) if notes else None

        return None, None, best_score, best_title, ";".join(notes) if notes else "no_open_pdf_found"

    def process(self, title: str, citing_papers: List[str]) -> DownloadResult:
        filename = safe_filename(title) + ".pdf"
        dest = self.outdir / filename

        if dest.exists() and not self.overwrite:
            return DownloadResult(
                title=title,
                citing_papers=citing_papers,
                status="exists",
                saved_path=str(dest),
                note="already_exists",
            )

        try:
            url, source, score, matched_title, note = self.resolve_one(title)
            if not url:
                return DownloadResult(
                    title=title,
                    citing_papers=citing_papers,
                    status="unresolved",
                    matched_title=matched_title,
                    score=score,
                    note=note,
                )

            self.download_pdf(url, dest)
            return DownloadResult(
                title=title,
                citing_papers=citing_papers,
                status="downloaded",
                source=source,
                matched_title=matched_title,
                score=score,
                download_url=url,
                saved_path=str(dest),
                note=note,
            )
        except Exception as exc:
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink(missing_ok=True)
            return DownloadResult(
                title=title,
                citing_papers=citing_papers,
                status="failed",
                note=str(exc),
            )
        finally:
            time.sleep(self.sleep_s)


def load_titles_from_json(json_path: Path) -> List[Tuple[str, List[str]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mapping = data.get("related_paper_to_citing_papers") or {}
    items = []
    for title, meta in mapping.items():
        citing_papers = meta.get("citing_papers") or []
        items.append((title, citing_papers))
    return items


def load_openreview_meta(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}

    def add_one(title: Optional[str], meta: Dict[str, Any]) -> None:
        if not title:
            return
        out[normalize_text(str(title))] = {
            "title": str(title),
            "openreview_id": str(meta.get("openreview_id") or "") if meta.get("openreview_id") else "",
            "openreview_url": str(meta.get("openreview_url") or "") if meta.get("openreview_url") else "",
            "pdf_url": str(meta.get("pdf_url") or "") if meta.get("pdf_url") else "",
        }

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            add_one(title, item)
    elif isinstance(data, dict):
        if "papers" in data and isinstance(data["papers"], list):
            for item in data["papers"]:
                if isinstance(item, dict):
                    add_one(item.get("title"), item)
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    add_one(value.get("title") or key, value)
    return out


def write_jsonl(path: Path, rows: Iterable[DownloadResult]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def summarize(results: List[DownloadResult]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in results:
        out[r.status] = out.get(r.status, 0) + 1
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download PDFs for titles in related_paper_to_citing_papers.json")
    p.add_argument("--input", required=True, help="Path to related_paper_to_citing_papers.json")
    p.add_argument("--outdir", required=True, help="Directory to save PDFs")
    p.add_argument("--log", default=None, help="Path to output JSONL log (default: <outdir>/download_log.jsonl)")
    p.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between papers")
    p.add_argument("--min-match-score", type=float, default=0.88, help="Minimum title similarity score")
    p.add_argument("--overwrite", action="store_true", help="Redownload even if file exists")
    p.add_argument(
        "--fallback-order",
        default="semantic_scholar,arxiv,openreview",
        help="Comma-separated resolvers, e.g. semantic_scholar,arxiv,openreview",
    )
    p.add_argument(
        "--openreview-meta",
        default=None,
        help="Optional JSON file containing title -> openreview_id/openreview_url metadata",
    )
    p.add_argument("--openreview-baseurl", default="https://api2.openreview.net")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--backoff-base", type=float, default=2.0)
    p.add_argument("--backoff-cap", type=float, default=60.0)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else (outdir / "download_log.jsonl")

    items = load_titles_from_json(input_path)
    if not items:
        print(f"No titles found in {input_path}", file=sys.stderr)
        return 1

    openreview_meta = load_openreview_meta(Path(args.openreview_meta)) if args.openreview_meta else {}

    downloader = PdfDownloader(
        outdir=outdir,
        min_match_score=args.min_match_score,
        sleep_s=args.sleep,
        overwrite=args.overwrite,
        s2_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        fallback_order=[x.strip() for x in args.fallback_order.split(",") if x.strip()],
        openreview_meta=openreview_meta,
        openreview_baseurl=args.openreview_baseurl,
        openreview_username=os.getenv("OPENREVIEW_USERNAME"),
        openreview_password=os.getenv("OPENREVIEW_PASSWORD"),
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
        backoff_cap=args.backoff_cap,
    )

    results: List[DownloadResult] = []
    total = len(items)
    print(f"Found {total} related paper titles.")

    for idx, (title, citing_papers) in enumerate(items, start=1):
        print(f"[{idx}/{total}] {title}")
        result = downloader.process(title, citing_papers)
        results.append(result)
        print(f"    -> {result.status} | source={result.source} | note={result.note}")
        write_jsonl(log_path, results)

    summary = summarize(results)
    print("\nSummary:")
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}")
    print(f"Log written to: {log_path}")
    print(f"PDF directory:   {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
