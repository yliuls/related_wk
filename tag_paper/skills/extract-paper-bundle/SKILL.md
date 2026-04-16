---
name: extract-paper-bundle
description: Read one target paper file once and produce both related_work.json and gold_annotations.json.
---

Input:
- paper_id: unique paper id
- paper_path: path to the target paper text file
- output_dir: directory for outputs

Read the target paper from `paper_path` once.

Task 1:
Write `output_dir/related_work.json`

Task 2:
Write `output_dir/gold_annotations.json`

Shared rules:
- Use only evidence from the target paper.
- Do not use external knowledge.
- Related works must come only from the paper's own cited works / references.
- Prefer precision over recall.
- Both outputs must be valid JSON.

For Task 1, use the related work taxonomy:
- `core_comparison_work`
- `method_lineage_or_bridge`
- `tension_or_conflict_work`

For Task 2, extract:
- `topic`
- `golden_gap`
- `golden_method`
- `gold_experiment`

The output schemas for the two files must match the project definitions.
