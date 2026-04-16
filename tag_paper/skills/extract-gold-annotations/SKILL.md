---
name: extract-gold-annotations
description: Extract topic, golden gap, golden method, and gold experiment from a given paper file path.
---

Input:
- paper_path: path to the target paper text file
- output_path: path to write gold_annotations.json

Read the target paper from `paper_path`.

Your task is to extract:
- `topic`
- `golden_gap`
- `golden_method`
- `gold_experiment`

Rules:
- Use only evidence from the target paper.
- Do not use external knowledge.
- Do not invent missing details.
- Prefer concise, evidence-grounded summaries.
- Write the result to `output_path`.

Definitions:
- `topic`: a continuous-text summary of the paper's central problem, task, domain, and setting
- `golden_gap`: a continuous-text summary of the main limitation, unmet need, unrealistic assumption, or evaluation problem that motivates the paper
- `golden_method`: a continuous-text summary of the proposed method at the right level of abstraction
- `gold_experiment`: a continuous-text summary of the main experimental design, datasets or benchmarks, comparisons, and evaluation focus

Return valid JSON only with this schema:

```json
{
  "target_paper_title": "string_or_null",
  "gold_annotations": {
    "topic": "string",
    "golden_gap": "string",
    "golden_method": "string",
    "gold_experiment": "string"
  }
}
```
