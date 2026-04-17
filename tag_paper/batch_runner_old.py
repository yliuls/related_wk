#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = DEFAULT_ROOT / "data" / "manifest.jsonl"
DEFAULT_LOG = DEFAULT_ROOT / "outputs" / "batch_runner_log.jsonl"
DEFAULT_BUNDLE_SKILL = DEFAULT_ROOT / "skills" / "extract-paper-bundle" / "SKILL.md"
DEFAULT_GOLD_SKILL = DEFAULT_ROOT / "skills" / "extract-gold-annotations" / "SKILL.md"
DEFAULT_RELATED_SKILL = DEFAULT_ROOT / "skills" / "extract-related-work" / "SKILL.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch paper annotation by invoking the local Codex CLI for each manifest item."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to manifest.jsonl.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based inclusive start index over manifest rows. Default: 1.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="1-based inclusive end index over manifest rows. Default: process to the end.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Only run the specified paper_id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if output files already exist.",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex CLI executable. Default: codex",
    )
    parser.add_argument(
        "--sandbox",
        default="workspace-write",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="Sandbox mode passed to codex exec.",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        default=True,
        help="Run each codex exec in an isolated ephemeral session. Default: enabled.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Codex model override.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional Codex profile override.",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=DEFAULT_LOG,
        help="Path to append batch runner logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without invoking Codex.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to run after filtering.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on line {lineno} of {path}")
            records.append(record)
    return records


def resolve_manifest_path(path_str: str, root_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def build_prompt(record: dict[str, Any], skill_path: Path) -> str:
    paper_id = record["paper_id"]
    paper_path = record["paper_path"]
    output_dir = record["output_dir"]
    return f"""Use the instructions in `{skill_path}`.

Process this single paper only.

Inputs:
- paper_id: {paper_id}
- paper_path: {paper_path}
- output_dir: {output_dir}

Required actions:
1. Read the target paper from `paper_path`.
2. Create `output_dir` if needed.
3. Write `output_dir/related_work.json`.
4. Write `output_dir/gold_annotations.json`.

Hard constraints:
- Use only evidence from the target paper.
- Do not use external knowledge.
- Related works must come only from the paper's own cited works / references.
- Both outputs must be valid JSON matching the project definitions.

When finished, briefly report whether both files were written successfully.
"""


def build_gold_prompt(record: dict[str, Any], skill_path: Path) -> str:
    paper_path = record["paper_path"]
    output_path = str(Path(record["output_dir"]) / "gold_annotations.json")
    return f"""Use the instructions in `{skill_path}`.

Process this single paper only.

Inputs:
- paper_path: {paper_path}
- output_path: {output_path}

Required actions:
1. Read the target paper from `paper_path`.
2. Create the parent directory for `output_path` if needed.
3. Write valid JSON to `output_path` using the skill schema.

Hard constraints:
- Use only evidence from the target paper.
- Do not use external knowledge.
- Output must be valid JSON matching the project definitions.

When finished, briefly report whether the file was written successfully.
"""


def build_related_prompt(record: dict[str, Any], skill_path: Path) -> str:
    paper_path = record["paper_path"]
    output_path = str(Path(record["output_dir"]) / "related_work.json")
    return f"""Use the instructions in `{skill_path}`.

Process this single paper only.

Inputs:
- paper_path: {paper_path}
- output_path: {output_path}

Required actions:
1. Read the target paper from `paper_path`.
2. Create the parent directory for `output_path` if needed.
3. Write valid JSON to `output_path` using the skill schema.

Hard constraints:
- Use only evidence from the target paper.
- Do not use external knowledge.
- Related works must come only from the paper's own cited works / references.
- Output must be valid JSON matching the project definitions.

When finished, briefly report whether the file was written successfully.
"""


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    end = args.end if args.end is not None else len(records)
    if args.start < 1:
        raise ValueError("--start must be >= 1")
    if end < args.start:
        raise ValueError("--end must be >= --start")

    indexed = list(enumerate(records, start=1))
    selected = [(idx, record) for idx, record in indexed if args.start <= idx <= end]

    if args.paper_id:
        wanted = set(args.paper_id)
        selected = [(idx, record) for idx, record in selected if record.get("paper_id") in wanted]

    if args.max_items is not None:
        selected = selected[: args.max_items]

    return selected


def append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_title_from_markdown(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Title:"):
                return line.split("Title:", 1)[1].strip()
    return path.stem


def get_expected_title(record: dict[str, Any], paper_path: Path) -> str:
    title = str(record.get("paper_title") or "").strip()
    if title:
        return title
    return read_title_from_markdown(paper_path)


def read_target_title_from_output(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    title = payload.get("target_paper_title")
    return str(title).strip() if title is not None else None


def needs_regeneration(path: Path, expected_title: str) -> bool:
    actual_title = read_target_title_from_output(path)
    if actual_title is None:
        return True
    return actual_title != expected_title


def determine_action(record: dict[str, Any], paper_path: Path, output_dir: Path) -> tuple[str, str]:
    expected_title = get_expected_title(record, paper_path)
    gold_path = output_dir / "gold_annotations.json"
    related_path = output_dir / "related_work.json"

    gold_bad = needs_regeneration(gold_path, expected_title)
    related_bad = needs_regeneration(related_path, expected_title)

    if gold_bad and related_bad:
        return "bundle", "missing or mismatched gold_annotations.json and related_work.json"
    if gold_bad:
        return "gold", "missing or mismatched gold_annotations.json"
    if related_bad:
        return "related", "missing or mismatched related_work.json"
    return "skip", "existing outputs match manifest paper_title"


def output_ok_for_action(action: str, output_dir: Path, expected_title: str) -> bool:
    gold_path = output_dir / "gold_annotations.json"
    related_path = output_dir / "related_work.json"
    if action == "bundle":
        return not needs_regeneration(gold_path, expected_title) and not needs_regeneration(related_path, expected_title)
    if action == "gold":
        return not needs_regeneration(gold_path, expected_title)
    if action == "related":
        return not needs_regeneration(related_path, expected_title)
    return True


def run_codex_exec(
    prompt: str,
    args: argparse.Namespace,
    root_dir: Path,
) -> tuple[int, float, str, str, str]:
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as temp_last:
        last_message_path = Path(temp_last.name)

    cmd = [
        args.codex_bin,
        "exec",
        "-C",
        str(root_dir),
        "--skip-git-repo-check",
        "--ephemeral",
        "--add-dir",
        str(root_dir),
        "-o",
        str(last_message_path),
        "-",
    ]
    if args.sandbox == "danger-full-access":
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    elif args.sandbox == "workspace-write":
        cmd.append("--full-auto")
    else:
        cmd.extend(["--sandbox", args.sandbox])
    if args.profile:
        cmd.extend(["--profile", args.profile])
    if args.model:
        cmd.extend(["--model", args.model])

    runner_home = root_dir / ".codex_batch_runner_home"
    runner_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(runner_home)
    env["CODEX_HOME"] = str(runner_home)
    env["XDG_CONFIG_HOME"] = str(runner_home / "xdg_config")
    env["XDG_CACHE_HOME"] = str(runner_home / "xdg_cache")
    env["XDG_STATE_HOME"] = str(runner_home / "xdg_state")
    env["XDG_DATA_HOME"] = str(runner_home / "xdg_data")
    env["TMPDIR"] = "/tmp"
    for key in [
        "HOME",
        "CODEX_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "XDG_DATA_HOME",
    ]:
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        duration = round(time.time() - started_at, 2)
        last_message = ""
        if last_message_path.exists():
            last_message = last_message_path.read_text(encoding="utf-8", errors="replace").strip()
        return proc.returncode, duration, proc.stdout, proc.stderr, last_message
    finally:
        if last_message_path.exists():
            last_message_path.unlink()


def run_one(
    idx: int,
    record: dict[str, Any],
    args: argparse.Namespace,
    root_dir: Path,
    bundle_skill_path: Path,
    gold_skill_path: Path,
    related_skill_path: Path,
) -> tuple[bool, str]:
    paper_id = str(record["paper_id"])
    paper_path = resolve_manifest_path(str(record["paper_path"]), root_dir)
    output_dir = resolve_manifest_path(str(record["output_dir"]), root_dir)

    if not paper_path.exists():
        return False, f"[{idx}] {paper_id}: paper file not found: {paper_path}"

    output_dir.mkdir(parents=True, exist_ok=True)

    expected_title = get_expected_title(record, paper_path)
    if args.force:
        action = "bundle"
        reason = "forced regeneration"
    else:
        action, reason = determine_action(record, paper_path, output_dir)

    if action == "skip":
        return True, f"[{idx}] {paper_id}: skipped, {reason}"

    if args.dry_run:
        return True, f"[{idx}] {paper_id}: dry-run, would run {action} because {reason}"

    if action == "bundle":
        prompt = build_prompt(record, bundle_skill_path)
    elif action == "gold":
        prompt = build_gold_prompt(record, gold_skill_path)
    else:
        prompt = build_related_prompt(record, related_skill_path)

    returncode, duration, stdout, stderr, last_message = run_codex_exec(
        prompt=prompt,
        args=args,
        root_dir=root_dir,
    )

    ok = returncode == 0 and output_ok_for_action(action, output_dir, expected_title)
    summary = (
        f"[{idx}] {paper_id}: "
        f"{'done' if ok else 'failed'} "
        f"via {action} (exit={returncode}, {duration}s)"
    )
    append_log(
        args.output_log,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "index": idx,
            "paper_id": paper_id,
            "paper_title": expected_title,
            "paper_path": str(paper_path),
            "output_dir": str(output_dir),
            "action": action,
            "reason": reason,
            "success": ok,
            "duration_sec": duration,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "last_message": last_message,
        },
    )
    if not ok and last_message:
        summary += f" | {last_message[:300]}"
    elif not ok:
        summary += " | outputs were not produced or title validation failed"
    return ok, summary


def main() -> int:
    args = parse_args()

    if shutil.which(args.codex_bin) is None:
        raise FileNotFoundError(f"Codex executable not found: {args.codex_bin}")

    manifest_path = args.manifest.expanduser().resolve()
    root_dir = manifest_path.parent.parent
    bundle_skill_path = DEFAULT_BUNDLE_SKILL
    gold_skill_path = DEFAULT_GOLD_SKILL
    related_skill_path = DEFAULT_RELATED_SKILL

    records = load_manifest(manifest_path)
    selected = select_records(records, args)

    if not selected:
        print("No manifest items selected.", flush=True)
        return 0

    success_count = 0
    failure_count = 0
    total = len(selected)

    print(f"Selected {total} items from manifest.", flush=True)

    for position, (idx, record) in enumerate(selected, start=1):
        paper_id = str(record.get("paper_id", f"item_{idx}"))
        paper_path = resolve_manifest_path(str(record["paper_path"]), root_dir)
        output_dir = resolve_manifest_path(str(record["output_dir"]), root_dir)
        if args.force:
            planned_action = "bundle"
            planned_reason = "forced regeneration"
        else:
            planned_action, planned_reason = determine_action(record, paper_path, output_dir)

        print(
            f"[{position}/{total}] {paper_id}: plan={planned_action} | reason={planned_reason}",
            flush=True,
        )

        ok, message = run_one(
            idx,
            record,
            args,
            root_dir=root_dir,
            bundle_skill_path=bundle_skill_path,
            gold_skill_path=gold_skill_path,
            related_skill_path=related_skill_path,
        )
        print(message, flush=True)
        if ok:
            success_count += 1
        else:
            failure_count += 1
        print(
            f"Progress: {position}/{total} finished | success={success_count} | failure={failure_count}",
            flush=True,
        )

    print(
        f"Finished {len(selected)} items. Success={success_count}, Failure={failure_count}",
        flush=True,
    )
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
