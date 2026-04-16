# AGENTS.md

## Repository Purpose

This directory runs batch target-paper annotation.

## Inputs

- Each target paper is stored as a separate Markdown file under [paper_content](/data3/yaofu/related_wk/tag_paper/paper_content).
- The batch manifest is stored in [data/manifest.jsonl](/data3/yaofu/related_wk/tag_paper/data/manifest.jsonl).

## Tasks

For each paper, produce:

- `related_work.json`
- `gold_annotations.json`

## Global Constraints

- Use only the target paper content.
- Do not use external knowledge.
- Related works must come only from the paper's own cited works / references.
- Outputs must be valid JSON.

## Output Locations

Write outputs to `outputs/<paper_id>/`.

## Batch Preference

- Prefer `skills/extract-paper-bundle/` for batch runs.
- Keep the single-task skills for debugging and ablation.
