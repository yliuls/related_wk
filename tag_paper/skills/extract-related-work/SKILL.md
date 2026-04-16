---
name: extract-related-work
description: Extract up to 10 target-paper-centric related works from a given paper file path.
---

Input:
- paper_path: path to the target paper text file
- output_path: path to write related_work.json

Read the target paper from `paper_path`.

Your task is to identify the most relevant related works from the paper's own cited works / references and write the result to `output_path`.

Rules:
- Select only from works explicitly cited in the target paper.
- Use only evidence from the target paper.
- Return at most 10 related works.
- Prefer precision over recall.
- Do not use external knowledge.
- Do not invent missing metadata.
- Order `related_work_items` from most important to least important.

Labels:
- `core_comparison_work`
- `method_lineage_or_bridge`
- `tension_or_conflict_work`

Label rules:
- Assign `core_comparison_work` if the work solves the same or a very similar core problem and is used as a main direct comparison target in experiments or performance discussion.
- Assign `method_lineage_or_bridge` if the work is a clear method source, inspiration, inherited technical line, or a bridge work that connects a similar method to a different problem or a similar problem to a different method.
- Assign `tension_or_conflict_work` if the work creates important tension with the target paper in assumptions, framing, conclusions, evaluation setting, or empirical findings, and helps explain why the target paper's gap or motivation is needed.

A work may receive multiple labels, but must have exactly one `primary_label`.

Return valid JSON only with this schema:

```json
{
  "target_paper_title": "string_or_null",
  "related_work_items": [
    {
      "citation_marker": "string_or_null",
      "paper_title": "string_or_null",
      "primary_label": "core_comparison_work | method_lineage_or_bridge | tension_or_conflict_work",
      "secondary_labels": ["core_comparison_work | method_lineage_or_bridge | tension_or_conflict_work"],
      "confidence": "high | medium | low",
      "evidence": [
        {
          "section": "string",
          "evidence_type": "explicit_claim | comparison_context | method_context | tension_context",
          "evidence_text": "string"
        }
      ],
      "gap_relevance_note": "string"
    }
  ]
}
```
