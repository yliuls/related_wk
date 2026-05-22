#!/usr/bin/env python3
"""Run LLM-based L3 topic grouping for L2 topic chunks."""
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

MAX_PAPERS_PER_L3_TOPIC = 25

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
3. Make topics fine-grained enough that each topic should contain at most 25 papers after assignment.
4. The topic set should be broad enough to cover every input paper.
5. Multi-label assignment will be allowed later, but do not assign papers now.
6. The number of topics should be appropriate for the input size."""

FINAL_PROMPT_TEMPLATE = """Step 3: optimize the final L3 topic set and assign papers.

You are given two independently proposed L3 topic sets. Use them as candidates, merge overlapping topics, split overly broad topics, remove weak topics, and produce the final topic assignment.

Candidate topic set A:

{proposal_1_json}

Candidate topic set B:

{proposal_2_json}

Final output rules:
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

2. Do not output id, paper_count, papers, topic_L2, or metadata. Python code will add those fields.
3. Each L3 topic must contain at most 25 paper_ids.
4. Every input paper_id must appear in at least one L3 topic.
5. A paper may appear in multiple L3 topics if it genuinely belongs to multiple fine-grained topics.
6. Do not invent paper IDs. Only use paper_id values from the input papers.
7. Within the same L3 topic, paper_ids must not contain duplicates.
8. If a semantic group would contain more than 25 papers, split it into more specific subtopics.

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


def build_final_messages(
    row: dict[str, Any],
    proposal_1: list[dict[str, str]],
    proposal_2: list[dict[str, str]],
) -> list[dict[str, str]]:
    final_prompt = FINAL_PROMPT_TEMPLATE.replace(
        "{proposal_1_json}",
        json.dumps({"topics_L3": proposal_1}, ensure_ascii=False),
    ).replace(
        "{proposal_2_json}",
        json.dumps({"topics_L3": proposal_2}, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        build_paper_context_message(row),
        {"role": "user", "content": final_prompt},
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


def normalize_topics(row: dict[str, Any], output: dict[str, Any]) -> list[dict[str, Any]]:
    topics = output.get("topics_L3")
    if not isinstance(topics, list) or not topics:
        raise ValueError("Missing or empty topics_L3 list")

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
        paper_ids_raw = topic.get("paper_ids")
        if not isinstance(paper_ids_raw, list) or not paper_ids_raw:
            raise ValueError(f"Topic {idx} has empty or invalid paper_ids")

        seen: set[str] = set()
        paper_items: list[dict[str, str]] = []
        for paper_id in paper_ids_raw:
            pid = str(paper_id).strip()
            if not pid or pid not in input_ids:
                raise ValueError(f"Topic {idx} invented paper_id {pid!r}")
            if pid in seen:
                raise ValueError(f"Topic {idx} duplicate paper_id {pid}")
            seen.add(pid)
            paper_items.append({"paper_id": pid, "title": title_by_id[pid]})

        if len(paper_items) > MAX_PAPERS_PER_L3_TOPIC:
            raise ValueError(f"Topic {idx} has {len(paper_items)} papers, exceeds {MAX_PAPERS_PER_L3_TOPIC}")

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
            "max_papers_per_topic": MAX_PAPERS_PER_L3_TOPIC,
            "direct_from_L2": False,
        },
        "topics_L3": topics_l3,
    }
    if final["source_line_index"] is None:
        final.pop("source_line_index")
    return final


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


def group_one_chunk(
    row: dict[str, Any],
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
    proposal_output_path: Path | None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        text: str | None = None
        current_stage = "proposal_1"
        try:
            current_stage = "proposal_1"
            proposal_1_obj, text, retry_after = request_and_parse_json(
                messages=build_proposal_messages(row, 1),
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
            proposal_1 = normalize_proposed_topics(proposal_1_obj)
            save_proposal_output(proposal_output_path, row, attempt=attempt, proposal_round=1, topics_l3=proposal_1, raw_output=text)

            current_stage = "proposal_2"
            proposal_2_obj, text, retry_after = request_and_parse_json(
                messages=build_proposal_messages(row, 2),
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
            proposal_2 = normalize_proposed_topics(proposal_2_obj)
            save_proposal_output(proposal_output_path, row, attempt=attempt, proposal_round=2, topics_l3=proposal_2, raw_output=text)

            current_stage = "final_assignment"
            final_obj, text, retry_after = request_and_parse_json(
                messages=build_final_messages(row, proposal_1, proposal_2),
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
            return assemble_final(row, normalize_topics(row, final_obj))
        except Exception as exc:  # noqa: BLE001 - CLI reports provider/validation errors.
            last_error = exc
            log_llm_validation_failure(
                validation_log_path,
                row,
                attempt=attempt,
                stage=current_stage,
                error=exc,
                raw_output=text,
            )
            if attempt == max_retries:
                break
            time.sleep(min(retry_sleep * (2 ** (attempt - 1)), 60.0))
    raise RuntimeError(f"Failed task={task_key_text(row)} after {max_retries} attempts: {last_error}")


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
    parser.add_argument("--output", default="data/output/l3_topics_from_llm.jsonl")
    parser.add_argument("--failed-output", default="data/output/l3_topics_failed.jsonl")
    parser.add_argument("--validation-log", default="data/output/l3_topics_validation_log.jsonl")
    parser.add_argument("--proposal-output", default="data/output/l3_topics_proposals.jsonl")
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
    failed_path = Path(args.failed_output)
    validation_log_path = Path(args.validation_log) if args.validation_log else None
    proposal_output_path = Path(args.proposal_output) if args.proposal_output else None
    completed = load_completed_keys(output_path) if args.resume else set()
    failed_keys = load_failed_keys(failed_path) if (args.resume and not args.retry_failed) else set()
    remaining = [r for r in rows if task_key(r) not in completed and task_key(r) not in failed_keys]
    if args.limit is not None:
        remaining = remaining[: args.limit]

    print(
        f"Loaded chunks {len(rows)}; completed {len(completed)}; skipped failed {len(failed_keys)}; "
        f"to process {len(remaining)}; concurrency {args.concurrency}"
    )
    if not remaining:
        return

    extra_body = parse_extra_body(args.extra_body_json)
    ok_count = 0
    fail_count = 0
    stop_requested = False

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_row = {
            executor.submit(
                group_one_chunk,
                row,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                extra_body=extra_body,
                validation_log_path=validation_log_path,
                proposal_output_path=proposal_output_path,
            ): row
            for row in remaining
        }
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            if stop_requested:
                continue
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                fail_count += 1
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
                print(f"FAILED {task_key_text(row)}: {exc}", file=sys.stderr)
                if args.stop_on_error:
                    stop_requested = True
                    for pending in future_to_row:
                        if not pending.done():
                            pending.cancel()
                    break
                continue

            append_jsonl_row(output_path, result)
            ok_count += 1
            print(f"Done {ok_count}/{len(remaining)}: {task_key_text(row)} -> {len(result['topics_L3'])} L3 topics")

    print(f"Wrote successes to {output_path}")
    if fail_count:
        print(f"Failed chunks: {fail_count}. See {failed_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
