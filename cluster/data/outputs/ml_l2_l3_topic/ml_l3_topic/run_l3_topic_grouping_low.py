#!/usr/bin/env python3
"""Run LLM-based L3 topic grouping for L2 topics split into chunks.

Pipeline:
  1. For every chunk under the same L2 topic, propose L3 topics twice.
  2. Optimize all chunk-level proposals into one shared L3 topic set per L2 topic.
  3. Assign papers in each chunk using only that optimized shared L3 topic set.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MAX_PAPERS_PER_ASSIGNMENT_REQUEST = 20

SYSTEM_PROMPT = """You are an expert research-paper topic curator for computer vision, machine learning, and AI conferences.

Your task is to group a set of papers from one coarse L2 topic into smaller, coherent L3 topics.

You will receive paper lists and step-specific instructions.

Always make semantic judgments from each paper's title and abstract.
Prefer grouping papers by their actual research problem, method family, data modality, or evaluation target, not by superficial keywords.

Return valid JSON only. Do not include markdown, comments, explanations, or extra text.

Topic names:
- "name" should be in Chinese.
- "name_en" should be in English.
- "description" should be in Chinese.

Avoid overly generic topic names such as "Other", "Miscellaneous", "General methods", or "Deep learning methods" unless absolutely necessary."""

PAPER_CONTEXT_PROMPT_TEMPLATE = """Input papers for one L2 topic chunk.

Input JSON:

{input_json}"""

PROPOSAL_PROMPT_TEMPLATE = """Step {proposal_round}: propose candidate fine-grained L3 topics for the input papers.

Only output topic definitions. Do not assign papers in this step.

Rules:
1. Output exactly one JSON object with this schema:

{
  "topics_L3": [
    {
      "name": "Chinese topic name",
      "name_en": "English topic name",
      "description": "A concise Chinese description of this L3 topic."
    }
  ]
}

2. Do not output paper_ids, papers, id, paper_count, topic_L2, or metadata.
3. Make topics fine-grained enough to be semantically useful after assignment.
4. The topic set should be broad enough to cover every input paper.
5. Multi-label assignment will be allowed later, but do not assign papers now.
6. The number of topics should be appropriate for the input size."""

OPTIMIZE_TOPIC_SET_PROMPT_TEMPLATE = """Step 3: optimize one shared final L3 topic set for the entire L2 topic.

You are given independently proposed L3 topic sets from all chunks of the same L2 topic. Use them as candidates, merge overlapping topics, split overly broad topics, remove weak topics, and produce one shared final L3 topic vocabulary for this L2 topic.

L2 topic:
{topic_l2}

Candidate L3 topic proposals from all chunks:

{all_proposals_json}

Rules:
1. Output exactly one JSON object with this schema:

{
  "topics_L3": [
    {
      "name": "Chinese topic name",
      "name_en": "English topic name",
      "description": "A concise Chinese description of this L3 topic."
    }
  ]
}

2. Do not output paper_ids, papers, id, paper_count, topic_L2, chunk_index, or metadata.
3. The topic set should cover the full L2 topic across all chunks.
4. Make topics fine-grained enough to be semantically useful per chunk after assignment.
5. Merge duplicate or near-duplicate candidate topics across chunks.
6. Split overly broad candidate topics into more specific topics when needed.
7. Preserve useful niche candidate topics even if they only appear in one chunk.

Return JSON only."""

ASSIGNMENT_PROMPT_TEMPLATE = """Step 4: assign papers in this chunk to the shared final L3 topic set.

Use only the final L3 topics below. Do not invent new L3 topics.

Final shared L3 topic set:

{final_topics_json}

Assignment output rules:
1. Output exactly one JSON object with this schema:

{
  "topics_L3": [
    {
      "name": "Chinese topic name",
      "name_en": "English topic name",
      "description": "A concise Chinese description of this L3 topic.",
      "paper_ids": ["00001", "00002"]
    }
  ]
}

2. Do not output id, paper_count, papers, topic_L2, chunk_index, or metadata. Python code will add those fields.
3. Every input paper_id must appear in at least one L3 topic.
4. A paper may appear in multiple L3 topics if it genuinely belongs to multiple fine-grained topics.
5. Do not invent paper IDs. Only use paper_id values from the input papers.
6. Treat paper_id as an exact string identifier. Copy it exactly from the input, including leading zeros. For example, output "00274", never "274".
7. Do not use ordinal positions, row numbers, or list indices as paper_ids.
8. Each output topic must match one item from the final shared L3 topic set by name, name_en, and description.

Return JSON only."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_input_chunk(row, line_no)
            rows.append(row)
    return rows


def validate_input_chunk(row: dict[str, Any], line_no: int) -> None:
    for field in ("topic", "paper_count", "papers", "chunk_index", "num_chunks", "source_paper_count"):
        if field not in row:
            raise ValueError(f"Line {line_no}: missing {field}")
    if not isinstance(row["papers"], list):
        raise ValueError(f"Line {line_no}: papers must be a list")
    if row["paper_count"] != len(row["papers"]):
        raise ValueError(f"Line {line_no}: paper_count mismatch")
    seen: set[str] = set()
    for paper in row["papers"]:
        for field in ("paper_id", "title", "abstract"):
            if not str(paper.get(field, "")).strip():
                raise ValueError(f"Line {line_no}: paper missing {field}")
        pid = str(paper["paper_id"])
        if pid in seen:
            raise ValueError(f"Line {line_no}: duplicate paper_id {pid}")
        seen.add(pid)


def task_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["topic"]), int(row["chunk_index"]), int(row["num_chunks"]))


def task_key_text(row: dict[str, Any]) -> str:
    return f"{row['topic']}|{row['chunk_index']}|{row['num_chunks']}"


def input_for_llm(row: dict[str, Any]) -> dict[str, Any]:
    return {"topic": row["topic"], "paper_count": row["paper_count"], "papers": row["papers"]}


def build_paper_context_message(row: dict[str, Any]) -> dict[str, str]:
    input_json = json.dumps(input_for_llm(row), ensure_ascii=False)
    return {"role": "user", "content": PAPER_CONTEXT_PROMPT_TEMPLATE.format(input_json=input_json)}


def build_proposal_messages(row: dict[str, Any], proposal_round: int) -> list[dict[str, str]]:
    proposal_prompt = PROPOSAL_PROMPT_TEMPLATE.replace("{proposal_round}", str(proposal_round))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        build_paper_context_message(row),
        {"role": "user", "content": proposal_prompt},
    ]


def build_optimization_messages(topic_l2: str, all_proposals: list[dict[str, Any]]) -> list[dict[str, str]]:
    optimize_prompt = OPTIMIZE_TOPIC_SET_PROMPT_TEMPLATE.replace(
        "{topic_l2}",
        topic_l2,
    ).replace(
        "{all_proposals_json}",
        json.dumps({"chunk_proposals": all_proposals}, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": optimize_prompt},
    ]


def build_assignment_messages(row: dict[str, Any], final_topics: list[dict[str, str]]) -> list[dict[str, str]]:
    assignment_prompt = ASSIGNMENT_PROMPT_TEMPLATE.replace(
        "{final_topics_json}",
        json.dumps({"topics_L3": final_topics}, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        build_paper_context_message(row),
        {"role": "user", "content": assignment_prompt},
    ]


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, str], str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - CLI accepts user-supplied endpoints.
            return resp.status, dict(resp.headers.items()), resp.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read().decode("utf-8", errors="replace")


def request_chat_once(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int,
    max_tokens: int | None,
    extra_body: dict[str, Any] | None,
) -> tuple[str, int | None]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra_body:
        payload.update(extra_body)

    status, resp_headers, text = post_json(url, headers, payload, timeout)
    if status == 400 and "response_format" in text:
        payload.pop("response_format", None)
        status, resp_headers, text = post_json(url, headers, payload, timeout)
    if status == 429:
        retry_after_raw = resp_headers.get("Retry-After") or resp_headers.get("retry-after")
        retry_after: int | None = None
        if retry_after_raw:
            try:
                retry_after = int(float(retry_after_raw))
            except ValueError:
                retry_after = None
        return "", retry_after
    if not (200 <= status < 300):
        raise RuntimeError(f"HTTP {status}: {text[:1000]}")
    data = json.loads(text)
    return data["choices"][0]["message"]["content"], None


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


def normalize_proposed_topics(output: dict[str, Any]) -> list[dict[str, str]]:
    topics = output.get("topics_L3")
    if not isinstance(topics, list) or not topics:
        raise ValueError("Missing or empty proposal topics_L3 list")

    normalized: list[dict[str, str]] = []
    for idx, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            raise ValueError(f"Proposal topic {idx} is not an object")
        for field in ("name", "name_en", "description"):
            if not str(topic.get(field, "")).strip():
                raise ValueError(f"Proposal topic {idx} missing {field}")
        normalized.append(
            {
                "name": str(topic["name"]).strip(),
                "name_en": str(topic["name_en"]).strip(),
                "description": str(topic["description"]).strip(),
            }
        )
    return normalized


def topic_identity(topic: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(topic.get("name", "")).strip(),
        str(topic.get("name_en", "")).strip(),
        str(topic.get("description", "")).strip(),
    )


def normalize_paper_id_from_llm(raw_paper_id: Any, input_ids: set[str]) -> str:
    pid = str(raw_paper_id).strip()
    if pid in input_ids:
        return pid
    if pid.isdigit():
        padded = pid.zfill(5)
        if padded in input_ids:
            return padded
    return pid


def normalize_paper_id_for_topic(raw_paper_id: Any, all_ids: set[str]) -> str:
    pid = str(raw_paper_id).strip()
    if pid in all_ids:
        return pid
    if pid.isdigit():
        padded = pid.zfill(5)
        if padded in all_ids:
            return padded
    return pid


def normalize_topics(
    row: dict[str, Any],
    output: dict[str, Any],
    allowed_topics: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    topics = output.get("topics_L3")
    if not isinstance(topics, list) or not topics:
        raise ValueError("Missing or empty topics_L3 list")

    allowed_topic_set = {topic_identity(topic) for topic in allowed_topics or []}
    title_by_id = {str(p["paper_id"]): str(p["title"]) for p in row["papers"]}
    input_ids = set(title_by_id)
    covered: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for idx, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            raise ValueError(f"Topic {idx} is not an object")
        for field in ("name", "name_en", "description", "paper_ids"):
            if field not in topic:
                raise ValueError(f"Topic {idx} missing {field}")
        if allowed_topic_set and topic_identity(topic) not in allowed_topic_set:
            raise ValueError(f"Topic {idx} is not in the optimized final L3 topic set: {topic_identity(topic)!r}")
        paper_ids_raw = topic.get("paper_ids")
        if not isinstance(paper_ids_raw, list) or not paper_ids_raw:
            raise ValueError(f"Topic {idx} has empty or invalid paper_ids")

        seen: set[str] = set()
        paper_items: list[dict[str, str]] = []
        for paper_id in paper_ids_raw:
            pid = normalize_paper_id_from_llm(paper_id, input_ids)
            if not pid or pid not in input_ids:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            paper_items.append({"paper_id": pid, "title": title_by_id[pid]})

        if not paper_items:
            continue

        covered.update(seen)
        normalized.append(
            {
                "id": f"T{idx:02d}",
                "name": str(topic["name"]).strip(),
                "name_en": str(topic["name_en"]).strip(),
                "description": str(topic["description"]).strip(),
                "paper_count": len(paper_items),
                "paper_ids": [p["paper_id"] for p in paper_items],
                "papers": paper_items,
            }
        )

    missing = input_ids - covered
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise ValueError(f"{len(missing)} input papers were not assigned to any L3 topic: {preview}")
    return normalized


def assemble_final(row: dict[str, Any], topics_l3: list[dict[str, Any]]) -> dict[str, Any]:
    final: dict[str, Any] = {
        "source_line_index": row.get("source_line_index"),
        "chunk_index": row["chunk_index"],
        "num_chunks": row["num_chunks"],
        "topic_L2": row["topic"],
            "metadata": {
                "total_papers": row["paper_count"],
                "num_topics": len(topics_l3),
                "multi_label": True,
                "max_papers_per_topic": None,
                "direct_from_L2": False,
            },
        "topics_L3": topics_l3,
    }
    if final["source_line_index"] is None:
        final.pop("source_line_index")
    return final


def split_assignment_rows(row: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    papers = row["papers"]
    batches = []
    for batch_index, start in enumerate(range(0, len(papers), batch_size), start=1):
        batch_papers = papers[start : start + batch_size]
        batch_row = dict(row)
        batch_row["papers"] = batch_papers
        batch_row["paper_count"] = len(batch_papers)
        batch_row["assignment_batch_index"] = batch_index
        batch_row["assignment_num_batches"] = (len(papers) + batch_size - 1) // batch_size
        batches.append(batch_row)
    return batches


def assemble_topic_final(
    topic_l2: str,
    rows: list[dict[str, Any]],
    assignment_results: list[dict[str, Any]],
    final_topics: list[dict[str, str]],
) -> dict[str, Any]:
    title_by_id = {
        str(paper["paper_id"]): str(paper["title"])
        for row in rows
        for paper in row["papers"]
    }
    all_ids = set(title_by_id)
    merged_ids_by_topic: dict[tuple[str, str, str], set[str]] = {
        topic_identity(topic): set()
        for topic in final_topics
    }
    topic_def_by_identity: dict[tuple[str, str, str], dict[str, str]] = {
        topic_identity(topic): topic
        for topic in final_topics
    }

    for result in assignment_results:
        for topic in result.get("topics_L3", []):
            identity = topic_identity(topic)
            if identity not in merged_ids_by_topic:
                continue
            for raw_pid in topic.get("paper_ids", []):
                pid = normalize_paper_id_for_topic(raw_pid, all_ids)
                if pid in all_ids:
                    merged_ids_by_topic[identity].add(pid)

    topics_l3: list[dict[str, Any]] = []
    for idx, identity in enumerate(topic_def_by_identity, start=1):
        paper_ids = sorted(merged_ids_by_topic[identity])
        if not paper_ids:
            continue
        topic = topic_def_by_identity[identity]
        papers = [{"paper_id": pid, "title": title_by_id[pid]} for pid in paper_ids]
        topics_l3.append(
            {
                "id": f"T{len(topics_l3) + 1:02d}",
                "name": topic["name"],
                "name_en": topic["name_en"],
                "description": topic["description"],
                "paper_count": len(paper_ids),
                "paper_ids": paper_ids,
                "papers": papers,
            }
        )

    assigned_ids = {
        paper_id
        for topic in topics_l3
        for paper_id in topic["paper_ids"]
    }
    missing = all_ids - assigned_ids
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise ValueError(f"{len(missing)} topic papers were not assigned to any L3 topic after merge: {preview}")

    return {
        "topic_L2": topic_l2,
        "metadata": {
            "total_papers": len(all_ids),
            "source_paper_count": rows[0]["source_paper_count"] if rows else 0,
            "num_chunks": len(rows),
            "chunk_indices": [int(row["chunk_index"]) for row in rows],
            "num_topics": len(topics_l3),
            "multi_label": True,
            "max_papers_per_assignment_request": MAX_PAPERS_PER_ASSIGNMENT_REQUEST,
            "direct_from_L2": False,
        },
        "topics_L3": topics_l3,
    }


def build_summary_from_outputs(output_path: Path) -> dict[str, Any]:
    topics_summary: dict[str, Any] = {}
    total_topics = 0
    total_l3_topics = 0
    total_assignments = 0

    if not output_path.exists():
        return {
            "output_path": str(output_path),
            "total_l2_topics": 0,
            "total_l3_topics": 0,
            "total_l3_paper_assignments": 0,
            "topics": {},
        }

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            topic_l2 = str(obj.get("topic_L2", ""))
            if not topic_l2:
                continue
            l3_counts = {
                str(topic.get("name_en") or topic.get("name")): int(topic.get("paper_count", 0))
                for topic in obj.get("topics_L3", [])
            }
            topics_summary[topic_l2] = {
                "total_papers": obj.get("metadata", {}).get("total_papers"),
                "num_l3_topics": len(l3_counts),
                "l3_topic_paper_counts": dict(sorted(l3_counts.items())),
            }

    total_topics = len(topics_summary)
    total_l3_topics = sum(v["num_l3_topics"] for v in topics_summary.values())
    total_assignments = sum(
        sum(v["l3_topic_paper_counts"].values())
        for v in topics_summary.values()
    )
    return {
        "output_path": str(output_path),
        "total_l2_topics": total_topics,
        "total_l3_topics": total_l3_topics,
        "total_l3_paper_assignments": total_assignments,
        "topics": dict(sorted(topics_summary.items())),
    }


def write_summary(output_path: Path, summary_path: Path | None) -> None:
    if summary_path is None:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(build_summary_from_outputs(output_path), f, ensure_ascii=False, indent=2)
        f.write("\n")


def log_llm_validation_failure(
    path: Path | None,
    row: dict[str, Any],
    *,
    attempt: int,
    stage: str,
    error: Exception,
    raw_output: str | None,
) -> None:
    if path is None:
        return
    log_row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topic": row["topic"],
        "source_line_index": row.get("source_line_index"),
        "chunk_index": row["chunk_index"],
        "num_chunks": row["num_chunks"],
        "source_paper_count": row["source_paper_count"],
        "paper_count": row["paper_count"],
        "attempt": attempt,
        "stage": stage,
        "error": repr(error),
        "raw_output": raw_output,
    }
    append_jsonl_row(path, log_row)


def save_proposal_output(
    path: Path | None,
    row: dict[str, Any],
    *,
    attempt: int,
    proposal_round: int,
    topics_l3: list[dict[str, str]],
    raw_output: str,
) -> None:
    if path is None:
        return
    proposal_row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topic": row["topic"],
        "source_line_index": row.get("source_line_index"),
        "chunk_index": row["chunk_index"],
        "num_chunks": row["num_chunks"],
        "source_paper_count": row["source_paper_count"],
        "paper_count": row["paper_count"],
        "attempt": attempt,
        "proposal_round": proposal_round,
        "topics_L3": topics_l3,
        "raw_output": raw_output,
    }
    append_jsonl_row(path, proposal_row)


def save_optimized_topic_set(
    path: Path | None,
    *,
    topic_l2: str,
    rows: list[dict[str, Any]],
    attempt: int,
    topics_l3: list[dict[str, str]],
    raw_output: str,
) -> None:
    if path is None:
        return
    optimized_row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topic": topic_l2,
        "num_chunks": len(rows),
        "source_paper_count": rows[0]["source_paper_count"] if rows else 0,
        "attempt": attempt,
        "topics_L3": topics_l3,
        "raw_output": raw_output,
    }
    append_jsonl_row(path, optimized_row)


def proposal_cache_key(
    topic: Any,
    chunk_index: Any,
    num_chunks: Any,
    proposal_round: Any,
) -> tuple[str, int, int, int]:
    return (str(topic), int(chunk_index), int(num_chunks), int(proposal_round))


def row_proposal_cache_key(row: dict[str, Any], proposal_round: int) -> tuple[str, int, int, int]:
    return proposal_cache_key(row["topic"], row["chunk_index"], row["num_chunks"], proposal_round)


def load_proposal_cache(path: Path | None) -> dict[tuple[str, int, int, int], list[dict[str, str]]]:
    if path is None or not path.exists():
        return {}

    cache: dict[tuple[str, int, int, int], list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                key = proposal_cache_key(
                    obj["topic"],
                    obj["chunk_index"],
                    obj["num_chunks"],
                    obj["proposal_round"],
                )
                cache[key] = normalize_proposed_topics(obj)
            except Exception as exc:  # noqa: BLE001 - cache may contain partial/bad rows from interrupted runs.
                print(f"Warning: skip unusable proposal cache line {line_no} in {path}: {exc}", file=sys.stderr)
                continue
    return cache


def load_optimized_topic_cache(path: Path | None) -> dict[str, list[dict[str, str]]]:
    if path is None or not path.exists():
        return {}

    cache: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                topic = str(obj["topic"])
                cache[topic] = normalize_proposed_topics(obj)
            except Exception as exc:  # noqa: BLE001 - cache may contain partial/bad rows from interrupted runs.
                print(f"Warning: skip unusable optimized topic cache line {line_no} in {path}: {exc}", file=sys.stderr)
                continue
    return cache


def request_and_parse_json(
    *,
    messages: list[dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: int,
    max_tokens: int | None,
    extra_body: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, int | None]:
    text, retry_after = request_chat_once(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    if retry_after is not None:
        return {}, text, retry_after
    return extract_json_object(text), text, None


def request_proposal_for_chunk(
    row: dict[str, Any],
    *,
    proposal_round: int,
    attempt: int,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: int,
    max_tokens: int | None,
    retry_sleep: float,
    extra_body: dict[str, Any] | None,
    validation_log_path: Path | None,
    proposal_output_path: Path | None,
) -> tuple[list[dict[str, str]] | None, bool]:
    text: str | None = None
    try:
        proposal_obj, text, retry_after = request_and_parse_json(
            messages=build_proposal_messages(row, proposal_round),
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        if retry_after is not None:
            time.sleep(max(retry_sleep, float(retry_after)))
            return None, True
        proposal = normalize_proposed_topics(proposal_obj)
        save_proposal_output(
            proposal_output_path,
            row,
            attempt=attempt,
            proposal_round=proposal_round,
            topics_l3=proposal,
            raw_output=text,
        )
        return proposal, False
    except Exception as exc:  # noqa: BLE001
        log_llm_validation_failure(
            validation_log_path,
            row,
            attempt=attempt,
            stage=f"proposal_{proposal_round}",
            error=exc,
            raw_output=text,
        )
        raise


def optimize_topic_set(
    topic_l2: str,
    rows: list[dict[str, Any]],
    all_proposals: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: int,
    max_tokens: int | None,
    max_retries: int,
    retry_sleep: float,
    extra_body: dict[str, Any] | None,
    optimized_topic_output_path: Path | None,
) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        text: str | None = None
        try:
            optimize_obj, text, retry_after = request_and_parse_json(
                messages=build_optimization_messages(topic_l2, all_proposals),
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            if retry_after is not None:
                time.sleep(max(retry_sleep, float(retry_after)))
                continue
            final_topics = normalize_proposed_topics(optimize_obj)
            save_optimized_topic_set(
                optimized_topic_output_path,
                topic_l2=topic_l2,
                rows=rows,
                attempt=attempt,
                topics_l3=final_topics,
                raw_output=text,
            )
            return final_topics
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(retry_sleep * (2 ** (attempt - 1)), 60.0))
    raise RuntimeError(f"Failed topic optimization for topic={topic_l2!r} after {max_retries} attempts: {last_error}")


def assign_one_chunk(
    row: dict[str, Any],
    final_topics: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: int,
    max_tokens: int | None,
    max_retries: int,
    retry_sleep: float,
    extra_body: dict[str, Any] | None,
    validation_log_path: Path | None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        text: str | None = None
        try:
            final_obj, text, retry_after = request_and_parse_json(
                messages=build_assignment_messages(row, final_topics),
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            if retry_after is not None:
                time.sleep(max(retry_sleep, float(retry_after)))
                continue
            return assemble_final(row, normalize_topics(row, final_obj, allowed_topics=final_topics))
        except Exception as exc:  # noqa: BLE001 - CLI reports provider/validation errors.
            last_error = exc
            log_llm_validation_failure(
                validation_log_path,
                row,
                attempt=attempt,
                stage="final_assignment",
                error=exc,
                raw_output=text,
            )
            if attempt == max_retries:
                break
            time.sleep(min(retry_sleep * (2 ** (attempt - 1)), 60.0))
    raise RuntimeError(f"Failed task={task_key_text(row)} after {max_retries} attempts: {last_error}")


def group_one_topic(
    topic_l2: str,
    rows: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: int,
    max_tokens: int | None,
    max_retries: int,
    retry_sleep: float,
    chunk_proposal_concurrency: int,
    extra_body: dict[str, Any] | None,
    validation_log_path: Path | None,
    proposal_output_path: Path | None,
    optimized_topic_output_path: Path | None,
    proposal_cache: dict[tuple[str, int, int, int], list[dict[str, str]]] | None,
    optimized_topic_cache: dict[str, list[dict[str, str]]] | None,
) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda r: int(r["chunk_index"]))
    all_proposals: list[dict[str, Any]] = []
    final_topics = (optimized_topic_cache or {}).get(topic_l2)
    print(f"[{topic_l2}] stage=optimized_cache {'hit' if final_topics is not None else 'miss'}", flush=True)

    if final_topics is None:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                proposals_by_key: dict[tuple[str, int, int, int], list[dict[str, str]]] = {}
                missing_tasks: list[tuple[dict[str, Any], int, tuple[str, int, int, int]]] = []

                for row in rows:
                    for proposal_round in (1, 2):
                        cache_key = row_proposal_cache_key(row, proposal_round)
                        cached_proposal = (proposal_cache or {}).get(cache_key)
                        if cached_proposal is not None:
                            proposals_by_key[cache_key] = cached_proposal
                        else:
                            missing_tasks.append((row, proposal_round, cache_key))

                cached_count = (len(rows) * 2) - len(missing_tasks)
                print(
                    f"[{topic_l2}] stage=proposal attempt={attempt} "
                    f"cache_hit={cached_count} missing={len(missing_tasks)}",
                    flush=True,
                )
                if missing_tasks:
                    with ThreadPoolExecutor(max_workers=chunk_proposal_concurrency) as executor:
                        future_to_task = {
                            executor.submit(
                                request_proposal_for_chunk,
                                row,
                                proposal_round=proposal_round,
                                attempt=attempt,
                                base_url=base_url,
                                api_key=api_key,
                                model=model,
                                temperature=temperature,
                                timeout=timeout,
                                max_tokens=max_tokens,
                                retry_sleep=retry_sleep,
                                extra_body=extra_body,
                                validation_log_path=validation_log_path,
                                proposal_output_path=proposal_output_path,
                            ): (row, proposal_round, cache_key)
                            for row, proposal_round, cache_key in missing_tasks
                        }
                        for future in as_completed(future_to_task):
                            row, proposal_round, cache_key = future_to_task[future]
                            print(
                                f"[{topic_l2}] stage=proposal_request_done "
                                f"chunk={row['chunk_index']}/{row['num_chunks']} "
                                f"round={proposal_round}",
                                flush=True,
                            )
                            proposal, retry = future.result()
                            if retry:
                                raise RuntimeError("Rate limited while proposing L3 topics")
                            if proposal is not None and proposal_cache is not None:
                                proposal_cache[cache_key] = proposal
                            if proposal is None:
                                raise RuntimeError(f"Empty proposal for {task_key_text(row)} round {proposal_round}")
                            proposals_by_key[cache_key] = proposal

                all_proposals = []
                for row in rows:
                    for proposal_round in (1, 2):
                        cache_key = row_proposal_cache_key(row, proposal_round)
                        proposal = proposals_by_key[cache_key]
                        all_proposals.append(
                            {
                                "chunk_index": row["chunk_index"],
                                "num_chunks": row["num_chunks"],
                                "proposal_round": proposal_round,
                                "topics_L3": proposal,
                            }
                        )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == max_retries:
                    raise RuntimeError(f"Failed proposal stage for topic={topic_l2!r}: {last_error}") from exc
                time.sleep(min(retry_sleep * (2 ** (attempt - 1)), 60.0))

        print(f"[{topic_l2}] stage=optimize_start proposals={len(all_proposals)}", flush=True)
        final_topics = optimize_topic_set(
            topic_l2,
            rows,
            all_proposals,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
            extra_body=extra_body,
            optimized_topic_output_path=optimized_topic_output_path,
        )
        if optimized_topic_cache is not None:
            optimized_topic_cache[topic_l2] = final_topics
        print(f"[{topic_l2}] stage=optimize_done topics_L3={len(final_topics)}", flush=True)
    else:
        print(f"[{topic_l2}] stage=optimize_skipped topics_L3={len(final_topics)}", flush=True)

    assignment_rows = [
        batch_row
        for row in rows
        for batch_row in split_assignment_rows(row, MAX_PAPERS_PER_ASSIGNMENT_REQUEST)
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=chunk_proposal_concurrency) as executor:
        future_to_row = {}
        for row in assignment_rows:
            print(
                f"[{topic_l2}] stage=assignment_start "
                f"chunk={row['chunk_index']}/{row['num_chunks']} "
                f"batch={row['assignment_batch_index']}/{row['assignment_num_batches']} "
                f"papers={row['paper_count']}",
                flush=True,
            )
            future = executor.submit(
                assign_one_chunk,
                row,
                final_topics,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                extra_body=extra_body,
                validation_log_path=validation_log_path,
            )
            future_to_row[future] = row

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            result = future.result()
            print(
                f"[{topic_l2}] stage=assignment_done "
                f"chunk={row['chunk_index']}/{row['num_chunks']} "
                f"batch={row['assignment_batch_index']}/{row['assignment_num_batches']} "
                f"topics_L3={len(result['topics_L3'])}",
                flush=True,
            )
            results.append(result)

    topic_result = assemble_topic_final(topic_l2, rows, results, final_topics)
    print(
        f"[{topic_l2}] stage=topic_merge_done "
        f"topics_L3={len(topic_result['topics_L3'])} "
        f"papers={topic_result['metadata']['total_papers']}",
        flush=True,
    )
    return topic_result


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    seen: set[tuple[str, int, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skip broken output line {line_no}: {exc}", file=sys.stderr)
                continue
            topic = obj.get("topic_L2")
            chunk_index = obj.get("chunk_index", 1)
            num_chunks = obj.get("num_chunks", 1)
            if topic is not None:
                metadata = obj.get("metadata", {})
                if "chunk_index" not in obj and isinstance(metadata, dict) and metadata.get("num_chunks"):
                    for idx in range(1, int(metadata["num_chunks"]) + 1):
                        seen.add((str(topic), idx, int(metadata["num_chunks"])))
                else:
                    seen.add((str(topic), int(chunk_index), int(num_chunks)))
    return seen


def load_failed_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    failed: set[tuple[str, int, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            failed.add((str(obj.get("topic", "")), int(obj.get("chunk_index", 1)), int(obj.get("num_chunks", 1))))
    return failed


def group_rows_by_topic(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["topic"]), []).append(row)
    return {
        topic: sorted(topic_rows, key=lambda r: int(r["chunk_index"]))
        for topic, topic_rows in sorted(grouped.items())
    }


def parse_extra_body(extra_body_json: str | None) -> dict[str, Any] | None:
    if not extra_body_json:
        return None
    data = json.loads(extra_body_json)
    if not isinstance(data, dict):
        raise ValueError("--extra-body-json must be a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/intermediate/l2_topics_split.jsonl")
    parser.add_argument("--output", default="data/output/l3_topics_by_topic.jsonl")
    parser.add_argument("--summary-output", default="data/output/l3_topics_summary.json")
    parser.add_argument("--failed-output", default="data/output/l3_topics_failed.jsonl")
    parser.add_argument("--validation-log", default="data/output/l3_topics_validation_log.jsonl")
    parser.add_argument("--proposal-output", default="data/output/l3_topics_proposals.jsonl")
    parser.add_argument(
        "--proposal-cache",
        default=None,
        help=(
            "Existing proposal JSONL to reuse. Defaults to --proposal-output. "
            "Use an empty string to disable proposal cache."
        ),
    )
    parser.add_argument("--optimized-topic-output", default="data/output/l3_topics_optimized_sets.jsonl")
    parser.add_argument(
        "--optimized-topic-cache",
        default=None,
        help=(
            "Existing optimized topic-set JSONL to reuse. Defaults to --optimized-topic-output. "
            "Use an empty string to disable optimized topic cache."
        ),
    )
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--extra-body-json",
        default=None,
        help='Optional JSON object merged into the LLM request body, e.g. \'{"reasoning_split": true}\'.',
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--save-every", type=int, default=20, help="Accepted for compatibility; successes are always appended immediately.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set environment variable {args.api_key_env}.")

    rows = read_jsonl(Path(args.input))
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else None
    failed_path = Path(args.failed_output)
    validation_log_path = Path(args.validation_log) if args.validation_log else None
    proposal_output_path = Path(args.proposal_output) if args.proposal_output else None
    optimized_topic_output_path = Path(args.optimized_topic_output) if args.optimized_topic_output else None
    if args.proposal_cache == "":
        proposal_cache_path = None
    elif args.proposal_cache is None:
        proposal_cache_path = proposal_output_path
    else:
        proposal_cache_path = Path(args.proposal_cache)
    proposal_cache = load_proposal_cache(proposal_cache_path)
    if args.optimized_topic_cache == "":
        optimized_topic_cache_path = None
    elif args.optimized_topic_cache is None:
        optimized_topic_cache_path = optimized_topic_output_path
    else:
        optimized_topic_cache_path = Path(args.optimized_topic_cache)
    optimized_topic_cache = load_optimized_topic_cache(optimized_topic_cache_path)
    completed = load_completed_keys(output_path) if args.resume else set()
    failed_keys = load_failed_keys(failed_path) if (args.resume and not args.retry_failed) else set()
    remaining = [r for r in rows if task_key(r) not in completed and task_key(r) not in failed_keys]
    if args.limit is not None:
        remaining = remaining[: args.limit]
    grouped_remaining = group_rows_by_topic(remaining)

    print(
        f"Loaded chunks {len(rows)}; completed {len(completed)}; skipped failed {len(failed_keys)}; "
        f"to process chunks {len(remaining)} across topics {len(grouped_remaining)}; "
        f"proposal cache entries {len(proposal_cache)}; "
        f"optimized topic cache entries {len(optimized_topic_cache)}; "
        f"topics run serially; per-topic chunk concurrency {args.concurrency}"
    )
    if not remaining:
        return

    extra_body = parse_extra_body(args.extra_body_json)
    ok_count = 0
    fail_count = 0

    for topic_l2, topic_rows in grouped_remaining.items():
        print(f"Start topic: {topic_l2} ({len(topic_rows)} chunks)")
        try:
            result = group_one_topic(
                topic_l2,
                topic_rows,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                chunk_proposal_concurrency=args.concurrency,
                extra_body=extra_body,
                validation_log_path=validation_log_path,
                proposal_output_path=proposal_output_path,
                optimized_topic_output_path=optimized_topic_output_path,
                proposal_cache=proposal_cache,
                optimized_topic_cache=optimized_topic_cache,
            )
        except Exception as exc:  # noqa: BLE001
            fail_count += len(topic_rows)
            for row in topic_rows:
                err_row = {
                    "topic": row["topic"],
                    "source_line_index": row.get("source_line_index"),
                    "chunk_index": row["chunk_index"],
                    "num_chunks": row["num_chunks"],
                    "source_paper_count": row["source_paper_count"],
                    "paper_count": row["paper_count"],
                    "error": repr(exc),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "input": input_for_llm(row),
                }
                append_jsonl_row(failed_path, err_row)
            print(f"FAILED topic={topic_l2}: {exc}", file=sys.stderr)
            if args.stop_on_error:
                break
            continue

        append_jsonl_row(output_path, result)
        write_summary(output_path, summary_path)
        ok_count += len(topic_rows)
        print(
            f"Done topic {ok_count}/{len(remaining)} chunks: "
            f"{result['topic_L2']} -> {len(result['topics_L3'])} L3 topics"
        )
        print(f"Finished topic: {topic_l2}")

    print(f"Wrote successes to {output_path}")
    write_summary(output_path, summary_path)
    if summary_path is not None:
        print(f"Wrote summary to {summary_path}")
    if fail_count:
        print(f"Failed chunks: {fail_count}. See {failed_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
