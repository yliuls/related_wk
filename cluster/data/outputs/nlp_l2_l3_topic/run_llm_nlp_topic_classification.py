#!/usr/bin/env python3
"""
Run LLM-based NLP canonical topic classification for a JSONL file.

Key behavior:
  - TOPICS is the single source of truth for the allowed topic list.
  - SYSTEM_PROMPT renders its "Allowed topic list" from TOPICS.
  - Each API request classifies exactly ONE paper.
  - --concurrency controls how many single-paper requests run in parallel.
  - The LLM input prompt contains only title + abstract, not paper_id.
  - The LLM output contains only topic, second_topic, selected_reason.
  - Python adds paper_id and title back when writing CSV/JSONL.
  - Each successful result is appended to JSONL immediately and fsync'ed.
  - CSV is regenerated every --checkpoint-every completed papers and at the end.
  - Failed papers are written to <output-prefix>.errors.jsonl; reruns resume from successes only.

Example:
  export LLM_API_KEY='sk-...'
  python run_llm_nlp_topic_classification.py \
    --input nlp_abstract_topic_canonical_none.jsonl \
    --output-prefix nlp_abstract_topic_none_llm \
    --base-url https://api.openai.com/v1 \
    --model gpt-5.5-thinking \
    --concurrency 8

For DeepSeek / MiniMax / OpenRouter, set --base-url and --model to their
OpenAI-compatible values.
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

# Keep the topic list synchronized with convert_topic.py.
from convert_topic import CANONICAL_TOPICS


TOPICS = list(CANONICAL_TOPICS)

if len(TOPICS) != len(set(TOPICS)):
    raise ValueError("Duplicate topic names found in TOPICS")

ALLOWED_TOPIC_LIST_TEXT = "\n".join(f"- {topic}" for topic in TOPICS)
TOPIC_SET = set(TOPICS)

SYSTEM_PROMPT = """You are an expert NLP conference area-chair assistant.

Your task is to classify exactly ONE NLP paper into the most appropriate canonical topic category using only its title and abstract.

You must make a semantic judgment from the paper's central research problem, contribution, method, task, and evaluation setting. Do NOT perform shallow keyword matching. Evidence phrases may be used only after the topic decision has been made.

Return exactly one JSON object with these fields:

```json
{
  "topic": "...",
  "second_topic": "...",
  "selected_reason": "..."
}
```

Rules:
1. `topic` must be exactly one item from the allowed topic list.
2. `second_topic` must be exactly one item from the allowed topic list, or an empty string if no meaningful secondary topic exists.
3. `selected_reason` must be a short, clear explanation of the reason for why the model is choosing the topic.
4. Do not output explanations, markdown, comments, or extra fields.
5. The primary topic should reflect the paper's central research problem, not merely the model architecture, dataset name, or incidental application.
6. If the paper proposes a general method evaluated across many tasks, classify by the dominant methodological contribution.
7. If the paper is primarily a dataset, benchmark, evaluation protocol, diagnostic suite, metric, or shared task, choose `Resources and Evaluation`.
8. If the paper has both a task domain and a method family, use the task/problem as `topic` and the method family as `second_topic` unless the method itself is the main contribution.
9. When uncertain, choose the topic that ACL/EMNLP/NAACL reviewers would most likely use to route the paper.

Allowed topic list:
""" + ALLOWED_TOPIC_LIST_TEXT + """

Selection priority:
- LLM safety, harmfulness, red-teaming, refusal behavior, preference alignment, RLHF/RLAIF, reward models for alignment -> `Safety and Alignment in LLMs`.
- Tool use, planning, autonomous or collaborative LLM systems, multi-agent workflows, agent memory, web or software agents -> `AI/LLM Agents`.
- Human-AI interaction, human feedback, annotation behavior, user studies, writing assistance, education-facing NLP -> `Human-Centered NLP`.
- Retrieval-augmented generation, knowledge-intensive QA with retrieval, dense/sparse retrieval for LMs, indexing and grounding by documents -> `Retrieval-Augmented Language Models`.
- Math, symbolic, logical, formal, theorem, program, or compositional reasoning -> `Mathematical, Symbolic, and Logical Reasoning in NLP`.
- Social media analysis, cultural analytics, misinformation, computational social science, NLP for social good -> `Computational Social Science, Cultural Analytics, and NLP for Social Good`.
- Code generation, code understanding, program repair, software engineering with language models -> `Code Models`.
- Interpretability, explainability, model editing, mechanistic analysis, transparency, attribution -> `Interpretability, Model Editing, Transparency, and Explainability`.
- Compression, quantization, pruning, distillation, efficient inference/training, low-resource methods -> `Efficient/Low-Resource Methods for NLP`.
- Domain adaptation, transfer, robustness, out-of-distribution generalization, continual learning -> `Generalizability and Transfer`.
- Dialogue systems, conversational agents, chatbots, interactive assistants -> `Dialogue and Interactive Systems`.
- Discourse, pragmatics, coreference, narrative structure, commonsense reasoning in context -> `Discourse, Pragmatics, and Reasoning`.
- Ethics, bias, fairness, privacy, copyright, governance, auditing social harms -> `Ethics, Bias, and Fairness`.
- Text generation, controlled generation, creative writing, data-to-text, style transfer -> `Natural Language Generation`.
- Information extraction, entity/relation/event extraction, retrieval, search, text mining -> `Information Extraction and Retrieval`.
- Language modeling, pretraining, instruction tuning, scaling, decoding, tokenization, reward/preference modeling as LM capability -> `Language Modeling`.
- General ML methods for NLP, optimization, training algorithms, architectures not better captured elsewhere -> `Machine Learning for NLP`.
- Linguistic theory, cognitive modeling, psycholinguistics, language acquisition, human language processing -> `Linguistic theories, Cognitive Modeling and Psycholinguistics`.
- Machine translation, speech translation, translation evaluation -> `Machine Translation`.
- Multilinguality, cross-lingual transfer, language diversity, low-resource languages, code-switching -> `Multilinguality and Language Diversity`.
- Vision-language, audio-language, robotics grounding, multimodal language understanding or generation -> `Multimodality and Language Grounding to Vision, Robotics and Beyond`.
- Neurosymbolic NLP or hybrid neural-symbolic approaches -> `Neurosymbolic approaches to NLP`.
- Phonology, morphology, word segmentation, subword or token segmentation with linguistic focus -> `Phonology, Morphology and Word Segmentation`.
- Question answering, reading comprehension, answer verification -> `Question Answering`.
- Datasets, benchmarks, evaluation metrics, shared tasks, annotation resources -> `Resources and Evaluation`.
- Semantics, lexical/sentence meaning, semantic parsing, textual inference, NLI -> `Semantics: Lexical, Sentence-level Semantics, Textual Inference and Other areas`.
- Sentiment, emotion, opinion mining, stance, argument mining, stylistic analysis -> `Sentiment Analysis, Stylistic Analysis, and Argument Mining`.
- ASR, TTS, speech processing, spoken language understanding -> `Speech Processing and Spoken Language Understanding`.
- Summarization, document or dialogue summarization -> `Summarization`.
- Syntax, parsing, tagging, chunking, hierarchical structure prediction -> `Hierarchical Structure Prediction, Syntax, and Parsing`.
- Practical NLP applications, domain applications such as biomedical, clinical, legal, education, finance, and other applied NLP not better captured by a specific topic -> `NLP Applications`.
"""

USER_PROMPT_TEMPLATE = """Classify the following paper.

title:
{title}

abstract:
{abstract}"""

CSV_FIELDS = ["paper_id", "title", "topic", "second_topic", "selected_reason"]


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
            rows.append({"paper_id": paper_id, "title": title, "abstract": abstract})
    return rows


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from a model response."""
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


def build_messages(paper: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=paper["title"],
        abstract=paper.get("abstract", ""),
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
            status = resp.status
            resp_headers = dict(resp.headers.items())
            text = resp.read().decode("utf-8")
            return status, resp_headers, text
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, dict(exc.headers.items()), text


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
    """Return response text and optional Retry-After seconds for rate-limit responses."""
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

    # Some OpenAI-compatible providers do not support response_format. Retry once without it on 400.
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
    """Validate model output and add paper_id/title from the input record."""
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


def classify_one_paper(
    paper: dict[str, Any],
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
) -> dict[str, str]:
    messages = build_messages(paper)
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
            return normalize_and_validate(paper, obj)
        except Exception as exc:  # noqa: BLE001 - CLI should report provider/validation errors clearly.
            last_error = exc
            if attempt == max_retries:
                break
            wait = min(retry_sleep * (2 ** (attempt - 1)), 60.0)
            time.sleep(wait)

    raise RuntimeError(f"Failed paper_id={paper['paper_id']} after {max_retries} attempts: {last_error}")


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
    """Append one JSONL row and force it to disk for robust resume."""
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
    parser.add_argument("--input", default="nlp_abstract_topic_canonical_none.jsonl")
    parser.add_argument("--output-prefix", default="nlp_abstract_topic_none_llm")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel single-paper API requests.")
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
        f"to process now {len(remaining)}; concurrency {args.concurrency}"
    )

    if not remaining:
        write_csv_from_jsonl(jsonl_path, csv_path)
        print(f"Nothing to do. Wrote {csv_path}")
        return

    completed = 0
    failed = 0
    stop_requested = False

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_paper = {
            executor.submit(
                classify_one_paper,
                paper,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                extra_body=extra_body,
            ): paper
            for paper in remaining
        }

        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            if stop_requested:
                continue

            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                err_row = {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "error": repr(exc),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                append_jsonl_row(errors_path, err_row)
                print(f"FAILED paper_id={paper['paper_id']}: {exc}", file=sys.stderr)
                print(f"Saved failure to {errors_path}", file=sys.stderr)

                if args.stop_on_error:
                    stop_requested = True
                    for pending in future_to_paper:
                        if not pending.done():
                            pending.cancel()
                    break
                continue

            # Save every successful paper immediately. Resume uses this JSONL file.
            append_jsonl_row(jsonl_path, row)
            completed += 1
            print(f"Done {completed}/{len(remaining)}: {row['paper_id']} -> {row['topic']} / {row['second_topic']}")

            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                write_csv_from_jsonl(jsonl_path, csv_path)
                print(f"Checkpoint: wrote {csv_path} after {completed} new results")

    write_csv_from_jsonl(jsonl_path, csv_path)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    if failed:
        print(f"Failed papers: {failed}. See {errors_path}. Re-run the same command to retry failed papers.", file=sys.stderr)


if __name__ == "__main__":
    main()
