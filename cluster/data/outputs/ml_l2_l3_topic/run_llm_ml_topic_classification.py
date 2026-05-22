#!/usr/bin/env python3
"""
Run LLM-based ML canonical topic classification for papers that need refinement.

Key behavior:
  - CANONICAL_TOPICS from convert_topic.py is the single source of truth.
  - SYSTEM_PROMPT renders the full canonical topic list.
  - --papers-per-request controls how many papers are included in one API request.
  - The user prompt includes title, abstract, official keywords, and llm_candidate_topics.
  - The LLM should prioritize llm_candidate_topics, but may choose from CANONICAL_TOPICS.
  - The LLM output contains results with paper_index, topic, second_topic, selected_reason.
  - Python adds paper_id and title back when writing CSV/JSONL.
  - Each successful result is appended to JSONL immediately and fsync'ed.
  - CSV is regenerated every --checkpoint-every completed papers and at the end.
  - Failed papers are written to <output-prefix>.errors.jsonl; reruns resume from successes only.

Example:
  export LLM_API_KEY='sk-...'
  python run_llm_ml_topic_classification.py \
    --input ml_topics_selected_fields_need_llm.jsonl \
    --output-prefix ml_topics_need_llm_classified \
    --base-url https://api.openai.com/v1 \
    --model gpt-5.5 \
    --concurrency 8 \
    --papers-per-request 1
"""
from __future__ import annotations

import argparse
import csv
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

from convert_topic import CANONICAL_TOPICS


TOPICS = list(CANONICAL_TOPICS)

if len(TOPICS) != len(set(TOPICS)):
    raise ValueError("Duplicate topic names found in CANONICAL_TOPICS")

ALLOWED_TOPIC_LIST_TEXT = "\n".join(f"- {topic}" for topic in TOPICS)
TOPIC_SET = set(TOPICS)

SYSTEM_PROMPT = """You are an expert machine learning conference area-chair assistant.

Your task is to classify one or more machine learning papers into the most appropriate canonical topic category.

Use the paper title, abstract, official keywords, and the provided candidate topics. Make a semantic judgment from the paper's central research problem, contribution, method, task, and evaluation setting. Do NOT perform shallow keyword matching.

Return exactly one JSON object with this shape:

```json
{
  "results": [
    {
      "paper_index": 1,
      "topic": "...",
      "second_topic": "...",
      "selected_reason": "..."
    }
  ]
}
```

Rules:
1. `topic` must be exactly one item from the allowed canonical topic list.
2. `second_topic` must be exactly one item from the allowed canonical topic list, or an empty string if no meaningful secondary topic exists.
3. Prioritize the user-provided `llm_candidate_topics` when they are semantically appropriate.
4. If none of the candidate topics fit well, choose the best topic from the full allowed canonical topic list.
5. The primary topic should reflect the paper's central research problem, not merely the model architecture, dataset name, or incidental application.
6. If the paper proposes a general method evaluated across many tasks, classify by the dominant methodological contribution.
7. If the paper is primarily a dataset, benchmark, evaluation protocol, diagnostic suite, metric, or reproducibility study, choose `Datasets, benchmarks, and evaluation`.
8. If the paper is mainly about LLMs or foundation models as systems or capabilities, choose `Foundation or frontier models, including LLMs`; use a task/application topic only when the application domain is the central contribution.
9. If the paper has both a task domain and a method family, use the central task/problem as `topic` and the method family as `second_topic` unless the method itself is clearly the main contribution.
10. `selected_reason` must be short and concrete.
11. Return exactly one result for every input paper.
12. `paper_index` must match the input paper_index.
13. Do not output explanations, markdown, comments, or extra fields.

Allowed canonical topic list:
""" + ALLOWED_TOPIC_LIST_TEXT

PAPER_PROMPT_TEMPLATE = """paper_index: {paper_index}

title:
{title}

abstract:
{abstract}

official_keywords:
{official_keywords}

llm_candidate_topics:
{llm_candidate_topics}
"""

USER_PROMPT_TEMPLATE = """Classify the following {paper_count} paper(s).

{papers_text}"""

CSV_FIELDS = ["paper_id", "title", "topic", "second_topic", "selected_reason"]


def format_list(value: Any) -> str:
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        return "\n".join(f"- {item}" for item in items) if items else "(none)"
    if value is None:
        return "(none)"
    text = str(value).strip()
    return text if text else "(none)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            paper_id = str(obj.get("paper_id", "")).strip()
            title = str(obj.get("title", "")).strip()
            abstract = str(obj.get("abstract", "")).strip()
            if not paper_id or not title:
                raise ValueError(f"Missing paper_id/title at line {line_no}")
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "official_keywords": obj.get("official_keywords") or [],
                    "llm_candidate_topics": obj.get("llm_candidate_topics") or [],
                }
            )
    return rows


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


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_messages(papers: list[dict[str, Any]]) -> list[dict[str, str]]:
    paper_prompts = []
    for index, paper in enumerate(papers, start=1):
        paper_prompts.append(
            PAPER_PROMPT_TEMPLATE.format(
                paper_index=index,
                title=paper["title"],
                abstract=paper.get("abstract", ""),
                official_keywords=format_list(paper.get("official_keywords")),
                llm_candidate_topics=format_list(paper.get("llm_candidate_topics")),
            )
        )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        paper_count=len(papers),
        papers_text="\n\n---\n\n".join(paper_prompts),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, str], str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user-supplied API endpoint for CLI use.
            return resp.status, dict(resp.headers.items()), resp.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read().decode("utf-8", errors="replace")


def request_chat_once(
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


def normalize_and_validate(paper: dict[str, Any], output: dict[str, Any]) -> dict[str, str]:
    pid = paper["paper_id"]
    topic = str(output.get("topic", "")).strip()
    second = str(output.get("second_topic", "")).strip()
    reason = str(output.get("selected_reason", "")).strip()

    if topic not in TOPIC_SET:
        raise ValueError(f"Invalid topic for {pid}: {topic!r}")
    if second and second not in TOPIC_SET:
        raise ValueError(f"Invalid second_topic for {pid}: {second!r}")
    if not reason:
        raise ValueError(f"Missing selected_reason for {pid}")

    return {
        "paper_id": pid,
        "title": paper["title"],
        "topic": topic,
        "second_topic": second,
        "selected_reason": reason,
    }


def normalize_and_validate_batch(
    papers: list[dict[str, Any]],
    output: dict[str, Any],
) -> list[dict[str, str]]:
    raw_results = output.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("LLM output must contain a results array")
    if len(raw_results) != len(papers):
        raise ValueError(f"Expected {len(papers)} results, got {len(raw_results)}")

    by_index: dict[int, dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError("Each result must be a JSON object")
        paper_index = int(item.get("paper_index", 0))
        if paper_index < 1 or paper_index > len(papers):
            raise ValueError(f"Invalid paper_index: {paper_index}")
        if paper_index in by_index:
            raise ValueError(f"Duplicate paper_index: {paper_index}")
        by_index[paper_index] = item

    rows = []
    for paper_index, paper in enumerate(papers, start=1):
        if paper_index not in by_index:
            raise ValueError(f"Missing result for paper_index={paper_index}")
        rows.append(normalize_and_validate(paper, by_index[paper_index]))
    return rows


def classify_paper_batch(
    papers: list[dict[str, Any]],
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
) -> list[dict[str, str]]:
    messages = build_messages(papers)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            text, retry_after = request_chat_once(
                base_url,
                api_key,
                model,
                messages,
                temperature,
                timeout,
                max_tokens,
                extra_body,
            )
            if retry_after is not None:
                time.sleep(max(retry_sleep, float(retry_after)))
                continue
            obj = extract_json_object(text)
            return normalize_and_validate_batch(papers, obj)
        except Exception as exc:  # noqa: BLE001 - CLI should report provider/validation errors clearly.
            last_error = exc
            if attempt == max_retries:
                break
            wait = min(retry_sleep * (2 ** (attempt - 1)), 60.0)
            time.sleep(wait)

    paper_ids = ",".join(paper["paper_id"] for paper in papers)
    raise RuntimeError(f"Failed paper_ids={paper_ids} after {max_retries} attempts: {last_error}")


def load_existing(jsonl_path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not jsonl_path.exists():
        return [], set()

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skip broken JSONL line {line_no} in {jsonl_path}: {exc}", file=sys.stderr)
                continue
            pid = str(obj.get("paper_id", "")).strip()
            if not pid or pid in seen:
                continue
            row = {
                "paper_id": pid,
                "title": str(obj.get("title", "")),
                "topic": str(obj.get("topic", "")),
                "second_topic": str(obj.get("second_topic", "")),
                "selected_reason": str(obj.get("selected_reason", "")),
            }
            rows.append(row)
            seen.add(pid)
    return rows, seen


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    final_rows, _ = load_existing(jsonl_path)
    final_rows.sort(key=lambda x: x["paper_id"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(final_rows)


def parse_extra_body(extra_body_json: str | None) -> dict[str, Any] | None:
    if not extra_body_json:
        return None
    data = json.loads(extra_body_json)
    if not isinstance(data, dict):
        raise ValueError("--extra-body-json must be a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ml_topics_selected_fields_need_llm.jsonl")
    parser.add_argument("--output-prefix", default="ml_topics_need_llm_classified")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel API requests.")
    parser.add_argument(
        "--papers-per-request",
        type=int,
        default=1,
        help="Number of papers to include in each API request. Default: 1.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Deprecated alias for --concurrency.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N papers after resume filtering.")
    parser.add_argument("--start-paper-id", default=None, help="Only process paper_id >= this value, e.g. 00101.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--extra-body-json", default=None, help='Optional JSON object merged into the API payload, e.g. \'{"thinking": false}\'.')
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Regenerate CSV every N successful new results. JSONL is still appended after every successful paper.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed paper. By default, failures are logged and other papers continue.",
    )
    args = parser.parse_args()

    if args.batch_size is not None:
        args.concurrency = args.batch_size
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.papers_per_request < 1:
        raise SystemExit("--papers-per-request must be >= 1")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set environment variable {args.api_key_env}.")

    input_path = Path(args.input)
    out_prefix = Path(args.output_prefix)
    jsonl_path = out_prefix.with_suffix(".jsonl")
    csv_path = out_prefix.with_suffix(".csv")
    errors_path = out_prefix.with_suffix(".errors.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    extra_body = parse_extra_body(args.extra_body_json)
    papers = read_jsonl(input_path)
    if args.start_paper_id is not None:
        papers = [p for p in papers if p["paper_id"] >= args.start_paper_id]

    _, seen = load_existing(jsonl_path)
    remaining = [p for p in papers if p["paper_id"] not in seen]
    if args.limit is not None:
        remaining = remaining[: args.limit]

    print(
        f"Loaded papers in scope {len(papers)}; already done {len(seen)}; "
        f"to process now {len(remaining)}; concurrency {args.concurrency}; "
        f"papers per request {args.papers_per_request}"
    )

    if not remaining:
        write_csv_from_jsonl(jsonl_path, csv_path)
        print(f"Nothing to do. Wrote {csv_path}")
        return

    completed = 0
    failed = 0
    stop_requested = False
    batches = chunked(remaining, args.papers_per_request)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_batch = {
            executor.submit(
                classify_paper_batch,
                batch,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                extra_body=extra_body,
            ): batch
            for batch in batches
        }

        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            if stop_requested:
                continue

            try:
                rows = future.result()
            except Exception as exc:  # noqa: BLE001
                failed += len(batch)
                for paper in batch:
                    err_row = {
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "error": repr(exc),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    append_jsonl_row(errors_path, err_row)
                paper_ids = ",".join(paper["paper_id"] for paper in batch)
                print(f"FAILED paper_ids={paper_ids}: {exc}", file=sys.stderr)
                print(f"Saved failure to {errors_path}", file=sys.stderr)

                if args.stop_on_error:
                    stop_requested = True
                    for pending in future_to_batch:
                        if not pending.done():
                            pending.cancel()
                    break
                continue

            for row in rows:
                append_jsonl_row(jsonl_path, row)
                completed += 1
                print(f"Done {completed}/{len(remaining)}: {row['paper_id']} -> {row['topic']} / {row['second_topic']}")

            if args.checkpoint_every > 0 and completed % args.checkpoint_every < len(rows):
                write_csv_from_jsonl(jsonl_path, csv_path)
                print(f"Checkpoint: wrote {csv_path} after {completed} new results")

    write_csv_from_jsonl(jsonl_path, csv_path)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    if failed:
        print(f"Failed papers: {failed}. See {errors_path}. Re-run the same command to retry failed papers.", file=sys.stderr)


if __name__ == "__main__":
    main()
