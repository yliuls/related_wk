# Tag Paper Batch Annotation

This directory is organized for batch target-paper annotation.

## Purpose

For each target paper, the project is meant to produce:

- `related_work.json`
- `gold_annotations.json`

The paper text used for extraction is stored as one Markdown file per paper under [paper_content](/data3/yaofu/related_wk/tag_paper/paper_content).

## Layout

```text
tag_paper/
├─ README.md
├─ AGENTS.md
├─ candidate_papers_os_ar.json
├─ download_and_parse_openreview.py
├─ json_to_paper_content.py
├─ generate_manifest.py
├─ paper_content/
│  └─ *.md
├─ data/
│  └─ manifest.jsonl
├─ outputs/
│  └─ <paper_id>/
│     ├─ related_work.json
│     └─ gold_annotations.json
├─ parsed_json/
├─ pdfs/
└─ skills/
   ├─ extract-related-work/
   │  └─ SKILL.md
   ├─ extract-gold-annotations/
   │  └─ SKILL.md
   └─ extract-paper-bundle/
      └─ SKILL.md
```

## Input Convention

- One paper per Markdown file
- Canonical paper content lives in [paper_content](/data3/yaofu/related_wk/tag_paper/paper_content)
- Use only the content of that paper file during extraction

## Batch Manifest

The manifest is stored at [data/manifest.jsonl](/data3/yaofu/related_wk/tag_paper/data/manifest.jsonl).

Each line has this shape:

```json
{"paper_id":"paper_0001","paper_path":"/abs/path/to/paper.md","output_dir":"/abs/path/to/outputs/paper_0001"}
```

Generate or refresh it with:

```bash
python /data3/yaofu/related_wk/tag_paper/generate_manifest.py
```

Run a batch slice with the local Codex CLI through [batch_runner.py](/data3/yaofu/related_wk/tag_paper/batch_runner.py):

```bash
python /data3/yaofu/related_wk/tag_paper/batch_runner.py \
  --manifest /data3/yaofu/related_wk/tag_paper/data/manifest.jsonl \
  --start 1 \
  --end 10
```

For a more stable launch, use [run_batch_runner.sh](/data3/yaofu/related_wk/tag_paper/run_batch_runner.sh). It first enters `tag_paper/` and then starts the runner:

```bash
bash /data3/yaofu/related_wk/tag_paper/run_batch_runner.sh safe --start 1 --end 10
```

If you want a more permissive Codex sandbox for child runs:

```bash
bash /data3/yaofu/related_wk/tag_paper/run_batch_runner.sh danger --start 1 --end 10
```

## Current Paper Preparation Flow

1. Download PDFs from OpenReview with [download_and_parse_openreview.py](/data3/yaofu/related_wk/tag_paper/download_and_parse_openreview.py)
2. Parse PDFs to structured JSON in `parsed_json/`
3. Convert parsed JSON to Markdown in `paper_content/` with [json_to_paper_content.py](/data3/yaofu/related_wk/tag_paper/json_to_paper_content.py)
4. Generate `data/manifest.jsonl`
5. Batch-run extraction using `skills/extract-paper-bundle/` or [batch_runner.py](/data3/yaofu/related_wk/tag_paper/batch_runner.py)

## JSON To Markdown Conversion

Use [json_to_paper_content.py](/data3/yaofu/related_wk/tag_paper/json_to_paper_content.py) to convert parsed paper JSON into Markdown files under [paper_content](/data3/yaofu/related_wk/tag_paper/paper_content).

Single-file conversion:

```bash
python /data3/yaofu/related_wk/tag_paper/json_to_paper_content.py \
  --input /data3/yaofu/related_wk/tag_paper/parsed_json/example.json \
  --output_dir /data3/yaofu/related_wk/tag_paper/paper_content
```

Batch conversion from a directory:

```bash
python /data3/yaofu/related_wk/tag_paper/json_to_paper_content.py \
  --input /data3/yaofu/related_wk/tag_paper/parsed_json \
  --output_dir /data3/yaofu/related_wk/tag_paper/paper_content
```

Behavior:

- If `--input` is a file, the script converts only that JSON file.
- If `--input` is a directory, the script converts all `.json` files in that directory.
- If the output directory already contains a Markdown file with the same basename, that file is skipped instead of overwritten.

## Batch Output Convention

For each paper:

- write [related_work.json](/data3/yaofu/related_wk/tag_paper/outputs) to `outputs/<paper_id>/related_work.json`
- write [gold_annotations.json](/data3/yaofu/related_wk/tag_paper/outputs) to `outputs/<paper_id>/gold_annotations.json`

## Principles

- Use only the target paper content
- Do not use external knowledge
- Related works must come only from the paper's own citations / references
- Prefer `extract-paper-bundle` for batch execution
