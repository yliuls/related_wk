from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .common import normalize_presentation_type, normalize_title


LOGGER = logging.getLogger(__name__)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "lxml")


def parse_cfp_topics_from_html(html: str) -> list[str]:
    """Extract CFP topic list from official CFP page."""
    if not html:
        return []
    try:
        soup = _soup(html)
        topics = []
        heading_patterns = re.compile(r"topics|topics of interest|subject areas|areas of interest", re.I)
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "strong", "p"]):
            if not heading_patterns.search(heading.get_text(" ", strip=True)):
                continue
            for sibling in heading.find_all_next(["ul", "ol"], limit=5):
                topics = _extract_topic_list_items(sibling)
                if topics:
                    break
            if topics:
                break
        if not topics:
            for candidate_list in soup.find_all(["ul", "ol"]):
                list_topics = _extract_topic_list_items(candidate_list)
                if sum(1 for item in list_topics if _looks_like_topic(item)) >= 3:
                    topics = list_topics
                    break
        return _dedupe(topics)
    except Exception as exc:
        LOGGER.warning("Failed to parse CFP topics: %s", exc)
        return []


def _looks_like_topic(text: str) -> bool:
    if not _is_valid_cfp_topic_candidate(text):
        return False
    lowered = text.lower()
    return bool(
        re.search(
            r"application|benchmark|dataset|evaluation|fairness|generation|infrastructure|language|learning|multimodal|"
            r"neuroscience|optimization|probabilistic|reasoning|retrieval|robotics|safety|speech|systems|theory|vision",
            lowered,
        )
    )


def _extract_topic_list_items(topic_list) -> list[str]:
    topics = []
    for li in topic_list.find_all("li", recursive=False):
        text = li.get_text(" ", strip=True)
        if _is_valid_cfp_topic_candidate(text):
            topics.append(text)
    return topics


def _is_valid_cfp_topic_candidate(text: str, max_length: int = 160) -> bool:
    if not text or len(text) > max_length or len(text) < 3:
        return False
    lowered = text.lower()
    non_topic_terms = (
        "deadline",
        "submission",
        "camera-ready",
        "registration",
        "author notification",
        "submit at",
        "openreview",
        "http",
    )
    return not any(term in lowered for term in non_topic_terms)


def parse_cvf_accepted_papers(html: str, venue: str, year: int) -> list[dict]:
    """Parse CVF accepted paper page metadata."""
    if not html:
        return []
    try:
        soup = _soup(html)
        records = []
        for dt in soup.find_all("dt", class_="ptitle"):
            title_link = dt.find("a")
            title = title_link.get_text(" ", strip=True) if title_link else dt.get_text(" ", strip=True)
            dd = dt.find_next_sibling("dd")
            authors = []
            if dd:
                authors = [a.get_text(" ", strip=True) for a in dd.find_all("a")] or _split_authors(dd.get_text(" ", strip=True))
            paper_url = title_link.get("href") if title_link else ""
            pdf_url = ""
            container_text = " ".join(
                node.get_text(" ", strip=True)
                for node in [dt, dt.find_next_sibling("dd"), dt.find_next_sibling("dd", class_="links")]
                if node
            )
            for link in dt.find_all_next("a", limit=8):
                href = link.get("href") or ""
                if "pdf" in href.lower():
                    pdf_url = href
                    break
            presentation = normalize_presentation_type(container_text)
            if presentation == "unknown":
                presentation = "poster"
            records.append(
                {
                    "paper_id": f"cvf:{venue.lower()}:{year}:{normalize_title(title)}",
                    "source_domain": "CV",
                    "venue": venue,
                    "year": year,
                    "title": title,
                    "authors": authors,
                    "paper_url": paper_url,
                    "pdf_url": pdf_url,
                    "paper_type": presentation,
                    "presentation_type": presentation,
                    "topic_status": "presentation_metadata_only",
                    "topic_confidence": 0.35,
                    "notes": "CV public page does not expose stable per-paper official topic; poster session is metadata, not topic.",
                    "raw_source": {"html_source": "cvf"},
                }
            )
        if records:
            return records

        for item in soup.select("li, article, .paper, .paper-card"):
            text = item.get_text(" ", strip=True)
            title_node = item.find(["a", "strong", "b"])
            title = title_node.get_text(" ", strip=True) if title_node else ""
            if not title or len(title) < 8:
                continue
            presentation = normalize_presentation_type(text)
            if presentation == "unknown":
                presentation = "poster"
            records.append(
                {
                    "paper_id": f"html:{venue.lower()}:{year}:{normalize_title(title)}",
                    "source_domain": "CV",
                    "venue": venue,
                    "year": year,
                    "title": title,
                    "authors": [],
                    "paper_url": title_node.get("href") if title_node and title_node.name == "a" else "",
                    "paper_type": presentation,
                    "presentation_type": presentation,
                    "topic_status": "presentation_metadata_only",
                    "topic_confidence": 0.35,
                    "raw_source": {"html_source": "accepted_page"},
                }
            )
        return records
    except Exception as exc:
        LOGGER.warning("Failed to parse CV accepted papers for %s %s: %s", venue, year, exc)
        return []


def parse_cvf_paper_detail(html: str) -> dict:
    """Parse metadata from a CVF paper detail page."""
    if not html:
        return {}
    try:
        soup = _soup(html)
        abstract_node = soup.find(id="abstract")
        abstract = abstract_node.get_text(" ", strip=True) if abstract_node else ""
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        title_meta = soup.find("meta", attrs={"name": "citation_title"})
        authors = [
            meta.get("content", "").strip()
            for meta in soup.find_all("meta", attrs={"name": "citation_author"})
            if meta.get("content", "").strip()
        ]
        return {
            "title": title_meta.get("content", "").strip() if title_meta else "",
            "authors": authors,
            "abstract": abstract,
            "pdf_url": pdf_meta.get("content", "").strip() if pdf_meta else "",
        }
    except Exception as exc:
        LOGGER.warning("Failed to parse CVF paper detail: %s", exc)
        return {}


def parse_acl_anthology_volume(html: str, venue: str, year: int) -> list[dict]:
    """Parse ACL Anthology main conference paper list."""
    if not html:
        return []
    try:
        soup = _soup(html)
        records = []
        for paper in soup.select("div.d-sm-flex.align-items-stretch.mb-3"):
            title_node = paper.select_one("span.d-block > strong a[href]")
            if not title_node:
                continue
            href = title_node.get("href") or ""
            paper_type = _acl_paper_type_from_href(href)
            if paper_type == "workshop":
                continue
            title = title_node.get_text(" ", strip=True).rstrip(".")
            if not title or _is_acl_volume_link(href):
                continue
            author_nodes = [
                author
                for author in paper.select("span.d-block > a[href^='/people/']")
                if author.get_text(" ", strip=True)
            ]
            authors = [author.get_text(" ", strip=True) for author in author_nodes]
            paper_url = urljoin("https://aclanthology.org", href)
            pdf_node = paper.select_one("a[aria-label='Open PDF'], a[href$='.pdf']")
            pdf_url = urljoin("https://aclanthology.org", pdf_node.get("href")) if pdf_node else ""
            abstract_node = paper.find_next_sibling("div", class_=re.compile(r"\babstract-collapse\b"))
            abstract = abstract_node.get_text(" ", strip=True) if abstract_node else ""
            records.append(
                {
                    "paper_id": f"acl:{venue.lower()}:{year}:{normalize_title(title)}",
                    "source_domain": "NLP",
                    "venue": venue,
                    "year": year,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "paper_url": paper_url,
                    "pdf_url": pdf_url,
                    "paper_type": paper_type,
                    "presentation_type": "unknown",
                    "is_best_paper": False,
                    "is_main_paper": paper_type == "main_paper",
                    "topic_status": "cfp_taxonomy_only",
                    "topic_confidence": 0.45,
                    "raw_source": {"html_source": "acl_anthology"},
                }
            )
        if records:
            return _dedupe_records(records)

        for paper in soup.select("p.d-sm-flex, .acl-paper, .paper"):
            title_node = paper.select_one("strong a, .title a, a.align-middle")
            if not title_node:
                continue
            title = title_node.get_text(" ", strip=True).rstrip(".")
            authors_node = paper.select_one(".text-muted, .authors")
            authors = _split_authors(authors_node.get_text(" ", strip=True)) if authors_node else []
            href = title_node.get("href") or ""
            paper_url = urljoin("https://aclanthology.org", href)
            records.append(
                {
                    "paper_id": f"acl:{venue.lower()}:{year}:{normalize_title(title)}",
                    "source_domain": "NLP",
                    "venue": venue,
                    "year": year,
                    "title": title,
                    "authors": authors,
                    "paper_url": paper_url,
                    "paper_type": "main_paper",
                    "presentation_type": "unknown",
                    "is_best_paper": False,
                    "is_main_paper": True,
                    "topic_status": "cfp_taxonomy_only",
                    "topic_confidence": 0.45,
                    "raw_source": {"html_source": "acl_anthology"},
                }
            )
        return records
    except Exception as exc:
        LOGGER.warning("Failed to parse ACL Anthology for %s %s: %s", venue, year, exc)
        return []


def parse_arr_area_keywords(html: str) -> list[dict]:
    """Parse ARR area keyword taxonomy from the official ARR areas page."""
    if not html:
        return []
    try:
        soup = _soup(html)
        areas = []
        for item in soup.find_all("li"):
            strong = item.find("strong")
            if not strong:
                continue
            area = strong.get_text(" ", strip=True)
            if not area:
                continue
            full_text = item.get_text(" ", strip=True)
            keyword_text = full_text.replace(area, "", 1).lstrip(" :")
            keywords = [part.strip(" .") for part in re.split(r";|,", keyword_text) if part.strip(" .")]
            if keywords:
                areas.append({"area": area, "keywords": keywords})
        return areas
    except Exception as exc:
        LOGGER.warning("Failed to parse ARR area keywords: %s", exc)
        return []


def _is_acl_volume_link(href: str) -> bool:
    return bool(re.search(r"/\d{4}\.[^.]+\.0/?$", href or ""))


def _acl_paper_type_from_href(href: str) -> str:
    value = href.strip("/").lower()
    if ".findings-" in value or "findings-" in value:
        return "findings"
    if "industry" in value:
        return "industry"
    if "demo" in value or "demos" in value:
        return "demo"
    if re.search(r"\.(acl|naacl)-(?:long|short)\.", value) or ".emnlp-main." in value:
        return "main_paper"
    return "workshop"


def parse_best_paper_page(html: str, venue: str, year: int) -> list[dict]:
    """Parse official best paper / awards page if available."""
    if not html:
        return []
    try:
        soup = _soup(html)
        records = []
        for node in soup.find_all(string=re.compile(r"best|outstanding|award", re.I)):
            container = node.parent
            for _ in range(3):
                if container and container.name not in ("li", "p", "article", "section", "div"):
                    container = container.parent
            if not container:
                continue
            text = container.get_text(" ", strip=True)
            title = _extract_quoted_title(text) or _extract_title_from_award_text(text)
            if not title:
                continue
            records.append(
                {
                    "paper_id": f"award:{venue.lower()}:{year}:{normalize_title(title)}",
                    "source_domain": "NLP",
                    "venue": venue,
                    "year": year,
                    "title": title,
                    "authors": [],
                    "paper_type": "best_paper",
                    "presentation_type": "unknown",
                    "is_best_paper": True,
                    "is_main_paper": True,
                    "topic_status": "cfp_taxonomy_only",
                    "topic_confidence": 0.55,
                    "raw_source": {"html_source": "awards_page"},
                }
            )
        return _dedupe_records(records)
    except Exception as exc:
        LOGGER.warning("Failed to parse best paper page for %s %s: %s", venue, year, exc)
        return []


def _split_authors(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^(authors?:)\s*", "", text, flags=re.I)
    parts = re.split(r"\s*,\s*|\s+and\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _extract_quoted_title(text: str) -> str | None:
    match = re.search(r"[\"“](.*?)[\"”]", text)
    return match.group(1).strip() if match else None


def _extract_title_from_award_text(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text or "").strip()
    match = re.search(r"(?:paper|award)[:\-]\s*(.+?)(?:\.| by |$)", text, re.I)
    if match:
        candidate = match.group(1).strip()
        if 8 <= len(candidate) <= 220:
            return candidate
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _dedupe_records(records: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for record in records:
        key = normalize_title(record.get("title", ""))
        if key and key not in seen:
            seen.add(key)
            output.append(record)
    return output
