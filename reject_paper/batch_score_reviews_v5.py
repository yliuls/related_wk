#!/usr/bin/env python3
"""
Batch-score paper review JSONL files with an LLM.

Version: dry-run-first-batch-prompt-txt + placeholder-title prompt + real-title-preserving output + resume.

Input:  JSONL, one paper per line, using the raw/flattened review schema.
Output: JSONL, one scored annotation per paper.

The script first compacts each raw paper record to only these fields:
- official_review.summary / strengths / weaknesses / questions
- reviewer_post_rebuttal_comments
- area_chair_evaluation.summary / reviewer_concerns / reviewer_scores
- final_decision.decision

Then it sends at most 5 papers per LLM call and writes one output JSON object per paper.

Resume behavior:
- The script writes completed batches to --output after every successful LLM call.
- With --resume, if --output already exists, the script counts existing valid JSONL rows,
  skips that many input papers, and continues from the next unfinished paper.
- Output writing is atomic, so interruption during saving should not corrupt the previous
  completed output file.

By default, the prompt sent to the LLM uses placeholder titles such as paper_1,
paper_2, etc. This avoids wasting tokens on long paper titles and prevents the
LLM from being responsible for copying titles correctly. The script still extracts
and stores the real title from the raw JSONL, then overwrites each final annotation's
paper_title with that real title before writing output.

Example:
  export OPENAI_API_KEY="..."
  python batch_score_reviews.py \
    --input reviews.jsonl \
    --output scored_reviews.jsonl \
    --model gpt-4.1-mini \
    --batch-size 5 \
    --pretty-output scored_reviews.pretty.json

Dry run, save the exact prompt for the first LLM batch to a txt file, then exit:
  python batch_score_reviews.py \
    --input reviews.jsonl \
    --output first_batch_prompt.txt \
    --dry-run

Resume after interruption:
  python batch_score_reviews.py \
    --input reviews.jsonl \
    --output scored_reviews.jsonl \
    --resume
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METHOD_DIMS = [
    "Consistency",
    "Clarity",
    "Novelty",
    "Soundness",
    "Feasibility",
    "Significance",
]

EXPERIMENT_DIMS = [
    "Consistency",
    "Soundness",
    "Completeness",
    "Feasibility",
    "Clarity_Reproducibility",
]

BATCH_PROMPT = r"""
You are an expert ML/NLP/CV conference review annotation assistant.

Your task is to analyze the official reviews, reviewer post-rebuttal comments, area chair summary, and final decision of submitted papers.

This is an annotation task. Do not simply summarize the reviews. Instead, map reviewer comments to predefined Method and Experiment evaluation dimensions.

You must output valid JSON only. Do not include markdown, explanations, or text outside the JSON.

You will receive a JSON object with key "papers", whose value is a list of at most 5 paper inputs.
Return a JSON object with key "papers", whose value is a list of annotation JSON objects in exactly the same order as the input papers.

# Scoring Scale

Use the following 4-point scale:

4 = excellent: The reviews clearly support or praise this dimension. There are few or no meaningful concerns.
3 = good: The reviews are mostly positive, with only minor concerns or requests for clarification.
2 = fair: The reviews recognize some value, but also raise important concerns.
1 = poor: The reviews identify serious problems that are unresolved or likely decision-critical.

If there is not enough evidence for a dimension, use null.

# Method Evaluation Dimensions

1. Consistency
- Does the method directly address the stated research gap or problem?
- Is the method aligned with the paper's core claim?

2. Clarity
- Is the method clearly explained?
- Are the algorithmic steps, formulas, assumptions, and implementation details understandable?

3. Novelty
- Does the method introduce a new idea, mechanism, theoretical view, or algorithmic design?
- Or is it mostly a minor modification or combination of prior work?

4. Soundness
- Is the method theoretically, mathematically, or logically valid?
- Are there concerns about correctness, convergence, bias, stability, or objective preservation?

5. Feasibility
- Can the method be implemented and scaled in realistic settings?
- Is the computational cost acceptable?

6. Significance
- Does the method address an important problem?
- Would the contribution matter to the target research community?

# Experiment Evaluation Dimensions

1. Consistency
- Do the experiments directly test the paper's main claims?
- Are the results aligned with what the method claims to improve?

2. Soundness
- Are the protocol, metrics, baselines, comparisons, and statistical analysis reliable?
- Are the baselines strong and fair?

3. Completeness
- Are the experiments sufficiently comprehensive?
- Are important datasets, tasks, baselines, ablations, sensitivity analyses, or robustness checks missing?

4. Feasibility
- Do the experiments show that the method can run in realistic settings?
- Do they report or imply reasonable runtime, scalability, and resource requirements?

5. Clarity_Reproducibility
- Are the experimental settings clearly described?
- Are datasets, hyperparameters, training/evaluation details, and implementation details sufficient for reproducibility?

# Annotation Rules

1. Use only evidence from the provided official reviews, reviewer post-rebuttal comments, area chair summary, and final decision.
2. Do not invent concerns or strengths that are not supported by the text.
3. The final annotation must consider rebuttal-stage information:
   - If reviewers changed their opinion after rebuttal, reflect the updated opinion.
   - If an initial concern was resolved or weakened after rebuttal, do not treat it as the main rejection reason unless the area chair or final decision still emphasizes it.
   - If a concern remains in the area chair summary or final decision, treat it as unresolved and decision-relevant.
4. When assigning scores, prioritize the area chair summary and final decision over individual reviewer comments.
5. If reviewers disagree, use the overall decision-relevant judgment rather than mechanically averaging reviewer opinions.
6. Distinguish between method and experiment issues:
   - Method issues include theoretical correctness, algorithm design, novelty, assumptions, convergence, stability, and feasibility of the proposed approach.
   - Experiment issues include baselines, metrics, datasets, protocols, ablations, reproducibility, scalability evidence, and empirical validation.
7. For rejection causes:
   - "category" must be one of: "Method", "Experiment", or "Other".
   - If a concern affects both method and experiment, choose the category that is more central to the rejection.
   - Use "Other" only for issues outside method or experiment, such as writing, scope mismatch, ethics, formatting, or venue fit.
   - "main_rejection_cause" should identify the most decision-critical issue.
   - "secondary_rejection_cause" should identify the next most important issue.
   - If there is no clear secondary cause, set category to "Other", dimension to "none", and reason to "No clear secondary rejection cause is supported by the reviews."

# Required JSON Output Schema for EACH paper

{
  "paper_title": "",
  "final_decision": "",
  "overall_summary": {
    "main_rejection_reason": "",
    "is_method_problem": true,
    "is_experiment_problem": true
  },
  "method_evaluation": [
    {"dimension": "Consistency", "score": null},
    {"dimension": "Clarity", "score": null},
    {"dimension": "Novelty", "score": null},
    {"dimension": "Soundness", "score": null},
    {"dimension": "Feasibility", "score": null},
    {"dimension": "Significance", "score": null}
  ],
  "experiment_evaluation": [
    {"dimension": "Consistency", "score": null},
    {"dimension": "Soundness", "score": null},
    {"dimension": "Completeness", "score": null},
    {"dimension": "Feasibility", "score": null},
    {"dimension": "Clarity_Reproducibility", "score": null}
  ],
  "main_rejection_cause": {
    "category": "Method",
    "dimension": "",
    "reason": ""
  },
  "secondary_rejection_cause": {
    "category": "Experiment",
    "dimension": "",
    "reason": ""
  },
  "final_one_sentence_conclusion": "This paper was mainly rejected because ..., not because ..."
}

Important constraints:
- For all scores, use only integer 1, 2, 3, 4, or null.
- overall_summary.is_method_problem and overall_summary.is_experiment_problem must be true, false, or null.
- Do not add extra top-level keys inside each annotation object.
- Return exactly this batch wrapper: {"papers": [ ... ]}
""".strip()


def textify(value: Any) -> str:
    """Convert OpenReview-style values into clean text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [textify(x) for x in value]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        for key in ("value", "text", "content", "description", "comment", "comments", "summary"):
            if key in value:
                return textify(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def deep_get(obj: Dict[str, Any], paths: List[List[str]], default: Any = None) -> Any:
    """Return the first existing nested value among several candidate paths."""
    for path in paths:
        cur: Any = obj
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur not in (None, ""):
            return cur
    return default


def get_by_dotted_path(record: Dict[str, Any], dotted_path: str) -> Any:
    """Read a user-specified dotted path such as submission.content.title.value."""
    cur: Any = record
    for part in dotted_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def extract_paper_title(
    record: Dict[str, Any], index: int, title_field: Optional[str] = None
) -> str:
    """Extract the real paper title if it exists; otherwise return an empty string.

    We intentionally do NOT fabricate titles such as paper_2, because that fake
    title would be passed to the LLM and may appear in the final annotation.
    The batch order is enough to align inputs and outputs.
    """
    if title_field:
        title_text = textify(get_by_dotted_path(record, title_field))
        if title_text:
            return title_text

    title = deep_get(
        record,
        [
            ["paper_title"],
            ["title"],
            ["submission", "title"],
            ["submission", "content", "title"],
            ["submission", "content", "title", "value"],
            ["submission_note", "title"],
            ["submission_note", "content", "title"],
            ["submission_note", "content", "title", "value"],
            ["paper", "title"],
            ["paper", "content", "title"],
            ["paper", "content", "title", "value"],
            ["note", "title"],
            ["note", "content", "title"],
            ["note", "content", "title", "value"],
            ["forum", "title"],
            ["forum", "content", "title"],
            ["forum", "content", "title", "value"],
            ["content", "title"],
            ["content", "title", "value"],
            ["metadata", "paper_title"],
            ["metadata", "title"],
            ["metadata", "content", "title"],
            ["metadata", "content", "title", "value"],
        ],
        default="",
    )
    return textify(title)


def normalize_post_rebuttal_comment(comment: Any) -> Dict[str, str]:
    if not isinstance(comment, dict):
        return {"comment": textify(comment)}

    reviewer_id = textify(
        comment.get("reviewer_id")
        or comment.get("reviewer_id_short")
        or comment.get("signature")
        or comment.get("signatures")
        or ""
    )
    comment_text = textify(
        comment.get("comment")
        or comment.get("comments")
        or comment.get("summary")
        or comment.get("reply")
        or comment.get("content")
        or comment.get("text")
        or comment.get("raw_content")
        or comment
    )
    result = {"comment": comment_text}
    if reviewer_id:
        result["reviewer_id"] = reviewer_id
    return result


def compact_paper(record: Dict[str, Any], index: int, title_field: Optional[str] = None) -> Dict[str, Any]:
    """Keep only the fields needed by the scoring prompt."""
    reviews_out: List[Dict[str, Any]] = []
    for r_i, review in enumerate(record.get("reviews") or []):
        if not isinstance(review, dict):
            continue
        official = review.get("official_review") or {}
        if not isinstance(official, dict):
            official = {}

        post_comments = review.get("reviewer_post_rebuttal_comments") or []
        if not isinstance(post_comments, list):
            post_comments = [post_comments]

        reviews_out.append(
            {
                "reviewer_id": textify(
                    review.get("reviewer_id_short")
                    or review.get("reviewer_id")
                    or f"reviewer_{r_i + 1}"
                ),
                "official_review": {
                    "summary": textify(official.get("summary")),
                    "strengths": textify(official.get("strengths")),
                    "weaknesses": textify(official.get("weaknesses")),
                    "questions": textify(official.get("questions")),
                },
                "reviewer_post_rebuttal_comments": [
                    normalize_post_rebuttal_comment(c) for c in post_comments if textify(c)
                ],
            }
        )

    ac = record.get("area_chair_evaluation") or {}
    if not isinstance(ac, dict):
        ac = {}

    final_decision = record.get("final_decision") or {}
    if not isinstance(final_decision, dict):
        final_decision = {"decision": final_decision}

    return {
        "paper_title": extract_paper_title(record, index, title_field=title_field),
        "reviews": reviews_out,
        "area_chair_evaluation": {
            "summary": textify(ac.get("summary")),
            "reviewer_concerns": textify(ac.get("reviewer_concerns")),
            "reviewer_scores": textify(ac.get("reviewer_scores")),
        },
        "final_decision": {
            "decision": textify(final_decision.get("decision")),
        },
    }


def read_jsonl(path: Path, limit: Optional[int] = None, start: int = 0) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no <= start:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            records.append(obj)
            if limit is not None and len(records) >= limit:
                break
    return records


def read_existing_output_prefix(path: Path) -> List[Dict[str, Any]]:
    """Read the valid completed prefix of an output JSONL file for --resume.

    If the last write was interrupted and left a partial/corrupt trailing line,
    this function keeps the valid prefix and ignores the corrupt suffix.
    """
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[warn] ignoring invalid/incomplete output JSONL from line {line_no} onward: {e}",
                    file=sys.stderr,
                )
                break
            if not isinstance(obj, dict):
                print(
                    f"[warn] ignoring non-object output JSONL from line {line_no} onward",
                    file=sys.stderr,
                )
                break
            records.append(obj)
    return records


def batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def parse_llm_json(text: str) -> Dict[str, Any]:
    """Parse JSON output. Also tolerates accidental code fences."""
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
        decoder = json.JSONDecoder()

        # Prefer likely wrapper starts. Reasoning models may emit prose containing
        # braces such as "{0, 1}" before the actual JSON object.
        starts = [
            m.start()
            for m in re.finditer(r'\{\s*"papers"\s*:', cleaned)
        ]
        starts.extend(m.start() for m in re.finditer(r"\{", cleaned))

        last_error: Optional[json.JSONDecodeError] = None
        for start in dict.fromkeys(starts):
            try:
                candidate, _ = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError as e:
                last_error = e
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get("papers"), list):
                parsed = candidate
                break

        if parsed is None:
            if last_error is not None:
                raise last_error
            raise
    if not isinstance(parsed, dict):
        raise ValueError("LLM output must be a JSON object with key 'papers'.")
    if "papers" not in parsed or not isinstance(parsed["papers"], list):
        raise ValueError("LLM output missing list key 'papers'.")
    return parsed


def normalize_scores(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Light cleanup to make downstream JSONL stable."""
    if not isinstance(annotation, dict):
        raise ValueError("Each annotation must be a JSON object.")

    def normalize_eval(items: Any, expected_dims: List[str]) -> List[Dict[str, Any]]:
        by_dim: Dict[str, Any] = {}
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    dim = item.get("dimension")
                    if dim in expected_dims:
                        score = item.get("score")
                        if score is not None:
                            try:
                                score = int(score)
                            except (TypeError, ValueError):
                                score = None
                            if score not in (1, 2, 3, 4):
                                score = None
                        by_dim[dim] = score
        return [{"dimension": dim, "score": by_dim.get(dim)} for dim in expected_dims]

    annotation["method_evaluation"] = normalize_eval(
        annotation.get("method_evaluation"), METHOD_DIMS
    )
    annotation["experiment_evaluation"] = normalize_eval(
        annotation.get("experiment_evaluation"), EXPERIMENT_DIMS
    )
    return annotation


def make_llm_batch(
    batch: List[Dict[str, Any]],
    *,
    title_mode: str = "placeholder",
    placeholder_start: int = 1,
) -> List[Dict[str, Any]]:
    """Return the paper batch that will actually be shown to the LLM.

    compact_paper() keeps the real paper_title for local bookkeeping.  The LLM
    does not need the real title for scoring, so by default we replace it with
    stable placeholders (paper_1, paper_2, ...).  Final output titles are later
    overwritten from the local real titles, not from the LLM output.
    """
    if title_mode not in {"placeholder", "real"}:
        raise ValueError("title_mode must be either 'placeholder' or 'real'.")

    llm_batch: List[Dict[str, Any]] = []
    for i, paper in enumerate(batch, start=placeholder_start):
        paper_for_llm = dict(paper)
        if title_mode == "placeholder":
            paper_for_llm["paper_title"] = f"paper_{i}"
        else:
            paper_for_llm["paper_title"] = textify(paper.get("paper_title"))
        llm_batch.append(paper_for_llm)
    return llm_batch


def build_messages(
    batch: List[Dict[str, Any]],
    *,
    title_mode: str = "placeholder",
    placeholder_start: int = 1,
) -> List[Dict[str, str]]:
    """Build the exact messages sent to the LLM for one batch."""
    user_payload = {
        "papers": make_llm_batch(
            batch, title_mode=title_mode, placeholder_start=placeholder_start
        )
    }
    return [
        {"role": "system", "content": BATCH_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def format_messages_for_inspection(messages: List[Dict[str, str]]) -> str:
    """Format LLM messages as a readable txt prompt-inspection file."""
    blocks: List[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        blocks.append(f"===== {role} MESSAGE =====\n{content}")
    return "\n\n".join(blocks).rstrip() + "\n"


def save_first_batch_prompt(
    path: Path,
    batch: List[Dict[str, Any]],
    *,
    title_mode: str = "placeholder",
) -> None:
    """Save the exact first-batch LLM prompt/messages to a txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    messages = build_messages(batch, title_mode=title_mode, placeholder_start=1)
    path.write_text(format_messages_for_inspection(messages), encoding="utf-8")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_error_log(
    path: Optional[Path],
    *,
    stage: str,
    error: BaseException,
    batch_source_indices: Optional[List[int]] = None,
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    raw_response: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if path is None:
        return
    record: Dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
    }
    if batch_source_indices is not None:
        record["batch_source_indices"] = batch_source_indices
    if attempt is not None:
        record["attempt"] = attempt
    if max_attempts is not None:
        record["max_attempts"] = max_attempts
    if model is not None:
        record["model"] = model
    if base_url is not None:
        record["base_url"] = base_url
    if messages is not None:
        record["messages"] = messages
    if raw_response is not None:
        record["raw_response"] = raw_response
    if extra:
        record["extra"] = extra
    append_jsonl(path, record)


def call_llm(
    batch: List[Dict[str, Any]],
    *,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    request_timeout: float,
    use_json_response_format: bool,
    title_mode: str,
    placeholder_start: int,
    error_log: Optional[Path],
    batch_source_indices: Optional[List[int]],
) -> List[Dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: openai. Install with `pip install openai`."
        ) from e

    client_kwargs: Dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    messages = build_messages(
        batch, title_mode=title_mode, placeholder_start=placeholder_start
    )

    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        content: Optional[str] = None
        try:
            params: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": request_timeout,
            }
            if use_json_response_format:
                params["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content or ""
            parsed = parse_llm_json(content)
            papers = parsed["papers"]
            if len(papers) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} annotations, got {len(papers)} annotations."
                )
            return [normalize_scores(p) for p in papers]
        except TypeError as e:
            # Some OpenAI-compatible providers do not support response_format/timeout.
            append_error_log(
                error_log,
                stage="llm_call",
                error=e,
                batch_source_indices=batch_source_indices,
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
                model=model,
                base_url=base_url,
                messages=messages,
                raw_response=content,
                extra={"use_json_response_format": use_json_response_format},
            )
            if use_json_response_format:
                use_json_response_format = False
                last_error = e
                continue
            last_error = e
        except Exception as e:  # noqa: BLE001 - CLI retries should catch provider errors.
            append_error_log(
                error_log,
                stage="llm_call",
                error=e,
                batch_source_indices=batch_source_indices,
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
                model=model,
                base_url=base_url,
                messages=messages,
                raw_response=content,
                extra={"use_json_response_format": use_json_response_format},
            )
            last_error = e

        if attempt < max_retries:
            sleep_s = min(2 ** attempt, 20)
            print(
                f"[warn] LLM call failed on attempt {attempt + 1}/{max_retries + 1}: {last_error}. "
                f"Retrying in {sleep_s}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed after retries: {last_error}")


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a text file to make resume safer after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    lines = [json.dumps(obj, ensure_ascii=False) for obj in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def write_pretty_json(path: Path, records: List[Dict[str, Any]]) -> None:
    atomic_write_text(path, json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score review JSONL files with an LLM, batching up to 5 papers per call."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input review JSONL path.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument(
        "--pretty-output",
        type=Path,
        default=None,
        help="Optional pretty JSON output path containing a list of all annotations.",
    )
    parser.add_argument(
        "--compact-output",
        type=Path,
        default=None,
        help="Optional JSONL path to save compacted LLM inputs for inspection.",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help=(
            "JSONL path for detailed failures, including batch input rows, traceback, "
            "LLM messages, and raw model response. Default: <output>.errors.jsonl."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save the exact first-batch LLM prompt/messages to --output as txt, then exit without calling the LLM.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run. If --output already exists, count its valid JSONL rows, "
            "skip that many input papers, and continue from the next unfinished paper."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of papers per LLM call. Must be between 1 and 5. Default: 5.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of papers to process.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip this many initial JSONL lines before processing. Default: 0.",
    )
    parser.add_argument(
        "--title-field",
        default=None,
        help=(
            "Optional dotted path for the real paper title in the raw JSONL, "
            "e.g. submission.content.title.value or content.title.value. "
            "Use this if your file stores titles in a custom location."
        ),
    )
    parser.add_argument(
        "--llm-title-mode",
        choices=["placeholder", "real"],
        default="placeholder",
        help=(
            "Which title to show in the LLM prompt. Default: placeholder, so the LLM sees "
            "paper_1/paper_2/... while final output is overwritten with the real extracted title. "
            "Use real only if you want the LLM to see real titles."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="Model name. Default: OPENAI_MODEL env var or gpt-4.1-mini.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key. Default: OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible base URL. Default: OPENAI_BASE_URL env var.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    # parser.add_argument("--max-tokens", type=int, default=196608)
    
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--no-json-response-format",
        action="store_true",
        help="Disable OpenAI JSON response_format for providers that do not support it.",
    )
    parser.add_argument(
        "--add-source-fields",
        action="store_true",
        help="Add source_index and source_paper_title to each output annotation for tracing.",
    )

    args = parser.parse_args()
    if args.error_log is None:
        args.error_log = args.output.with_name(args.output.stem + ".errors.jsonl")
    print(f"[info] detailed errors will be appended to {args.error_log}", file=sys.stderr)

    if not 1 <= args.batch_size <= 5:
        raise SystemExit("--batch-size must be between 1 and 5.")

    raw_records = read_jsonl(args.input, limit=args.limit, start=args.start)
    source_indices = [args.start + i + 1 for i in range(len(raw_records))]
    compact_records = [
        compact_paper(r, index=source_indices[i], title_field=args.title_field)
        for i, r in enumerate(raw_records)
    ]

    missing_title_indices = [
        source_indices[i]
        for i, paper in enumerate(compact_records)
        if not paper.get("paper_title")
    ]
    if missing_title_indices:
        preview = ", ".join(map(str, missing_title_indices[:10]))
        more = "..." if len(missing_title_indices) > 10 else ""
        print(
            f"[warn] no real paper title found for JSONL line(s): {preview}{more}. "
            "The prompt/final output will use an empty paper_title for those papers. "
            "Pass --title-field if your title is stored in a custom path.",
            file=sys.stderr,
        )

    if args.compact_output:
        write_jsonl(args.compact_output, compact_records)
        print(f"[ok] wrote compacted inputs to {args.compact_output}", file=sys.stderr)

    if args.dry_run:
        first_batch = compact_records[: args.batch_size]
        save_first_batch_prompt(args.output, first_batch, title_mode=args.llm_title_mode)
        print(
            f"[ok] dry run wrote first-batch LLM prompt for {len(first_batch)} paper(s) to {args.output}",
            file=sys.stderr,
        )
        return 0

    existing_annotations: List[Dict[str, Any]] = []
    resume_offset = 0
    if args.resume:
        existing_annotations = read_existing_output_prefix(args.output)
        resume_offset = len(existing_annotations)
        if resume_offset > len(compact_records):
            raise SystemExit(
                f"--resume found {resume_offset} existing output rows, but only "
                f"{len(compact_records)} input papers are selected. Check --input/--limit/--start/--output."
            )
        if resume_offset:
            print(
                f"[resume] found {resume_offset} completed annotation(s) in {args.output}; "
                f"skipping the first {resume_offset} selected input paper(s).",
                file=sys.stderr,
            )
        if resume_offset == len(compact_records):
            print(
                f"[resume] all {resume_offset} selected paper(s) already completed; nothing to do.",
                file=sys.stderr,
            )
            if args.pretty_output:
                write_pretty_json(args.pretty_output, existing_annotations)
                print(f"[ok] wrote pretty JSON to {args.pretty_output}", file=sys.stderr)
            return 0

    if not args.api_key and not args.base_url:
        # Some local OpenAI-compatible servers do not need an API key but do need base_url.
        raise SystemExit(
            "Missing API config. Set OPENAI_API_KEY, or pass --api-key/--base-url for an OpenAI-compatible endpoint."
        )

    all_annotations: List[Dict[str, Any]] = list(existing_annotations)
    remaining_total = len(compact_records) - resume_offset
    for batch_i, start_i in enumerate(
        range(resume_offset, len(compact_records), args.batch_size), start=1
    ):
        batch = compact_records[start_i : start_i + args.batch_size]
        batch_source_indices = source_indices[start_i : start_i + args.batch_size]
        print(
            f"[info] scoring batch {batch_i}: input rows {batch_source_indices[0]}-"
            f"{batch_source_indices[-1]} ({len(batch)} paper(s)); "
            f"{remaining_total - (start_i - resume_offset)} paper(s) remaining before this batch",
            file=sys.stderr,
        )
        annotations = call_llm(
            batch,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            use_json_response_format=not args.no_json_response_format,
            title_mode=args.llm_title_mode,
            placeholder_start=1,
            error_log=args.error_log,
            batch_source_indices=batch_source_indices,
        )
        # Keep stable metadata from the input instead of trusting the LLM to copy it.
        for ann, src in zip(annotations, batch):
            ann["paper_title"] = src.get("paper_title", "")
            src_decision = (src.get("final_decision") or {}).get("decision", "")
            if src_decision:
                ann["final_decision"] = src_decision

        if args.add_source_fields:
            for ann, src, source_index in zip(annotations, batch, batch_source_indices):
                ann["source_index"] = source_index
                ann["source_paper_title"] = src.get("paper_title")
        all_annotations.extend(annotations)

        # Write incrementally so completed batches are not lost if a later batch fails.
        # The write itself is atomic; with --resume, the next run skips these rows.
        write_jsonl(args.output, all_annotations)
        if args.pretty_output:
            write_pretty_json(args.pretty_output, all_annotations)
        print(
            f"[ok] saved progress: {len(all_annotations)}/{len(compact_records)} annotation(s)",
            file=sys.stderr,
        )

    print(f"[ok] wrote {len(all_annotations)} annotations to {args.output}", file=sys.stderr)
    if args.pretty_output:
        print(f"[ok] wrote pretty JSON to {args.pretty_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
