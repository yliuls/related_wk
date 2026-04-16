# Target Paper Batch Annotation Project Design for Codex

## 目标

这个项目用于**批量处理很多篇 target paper**，并从每篇论文中提取两类结构化结果：

1. `related_work.json`
2. `gold_annotations.json`

其中：

- `related_work.json`：从 target paper 的**引用 / reference** 中，选出最多 10 篇最相关的 related work，并打上角色标签。
- `gold_annotations.json`：从 target paper 正文中抽取：
  - `topic`
  - `golden_gap`
  - `golden_method`
  - `gold_experiment`

本项目后续会用于**批量跑论文提取任务**，因此仓库设计不能依赖把单篇论文内容放进 `README.md`。

---

## 仓库应改成的形态

建议把仓库整理成下面这种结构：

```text
project/
├─ README.md
├─ AGENTS.md
├─ data/
│  ├─ papers/
│  │  ├─ paper_0001.txt
│  │  ├─ paper_0002.txt
│  │  └─ ...
│  └─ manifest.jsonl
├─ outputs/
│  ├─ paper_0001/
│  │  ├─ related_work.json
│  │  └─ gold_annotations.json
│  ├─ paper_0002/
│  │  ├─ related_work.json
│  │  └─ gold_annotations.json
│  └─ ...
└─ skills/
   ├─ extract-related-work/
   │  └─ SKILL.md
   ├─ extract-gold-annotations/
   │  └─ SKILL.md
   └─ extract-paper-bundle/
      └─ SKILL.md
```

### 各部分职责

#### `README.md`
只放项目说明，不放 target paper 内容。

#### `AGENTS.md`
放仓库级的固定规则，比如：

- 每篇论文单独存放在 `data/papers/`
- 只能根据 target paper 内容抽取
- 不允许使用外部知识
- related work 只能从论文自己的 cited works / references 中选择
- 输出路径固定在 `outputs/<paper_id>/`

#### `data/papers/`
每篇 target paper 一份文本文件。
建议是清洗后的论文正文，尽量保留：

- 标题
- 摘要
- Introduction
- Related Work
- Method
- Experiments
- Conclusion
- References

#### `data/manifest.jsonl`
用于批量跑。每行描述一篇论文：

```json
{"paper_id":"paper_0001","paper_path":"data/papers/paper_0001.txt","output_dir":"outputs/paper_0001"}
{"paper_id":"paper_0002","paper_path":"data/papers/paper_0002.txt","output_dir":"outputs/paper_0002"}
```

#### `outputs/`
每篇论文一个目录，分别存两个结果文件：

- `related_work.json`
- `gold_annotations.json`

#### `skills/`
存三个 skill：

1. `extract-related-work`
2. `extract-gold-annotations`
3. `extract-paper-bundle`

---

## 为什么用三个 skill

### `extract-related-work`
单任务 skill，只做：

- 读取一篇 target paper
- 从该论文自己的引用 / references 中选出最多 10 篇 related work
- 输出 `related_work.json`

### `extract-gold-annotations`
单任务 skill，只做：

- 读取一篇 target paper
- 抽取 `topic / golden_gap / golden_method / gold_experiment`
- 输出 `gold_annotations.json`

### `extract-paper-bundle`
总控 skill，一次完成两个任务：

- 只读取一次 target paper
- 同时写出：
  - `related_work.json`
  - `gold_annotations.json`

这个 skill 最适合后面批量跑 many papers，因为它避免同一篇论文被重复读取两次。

---

## 推荐的 AGENTS.md 目标内容

Codex 应该在项目根目录创建一个简洁的 `AGENTS.md`，内容重点是仓库级规则，而不是塞很长的 task prompt。

建议内容如下：

```md
# AGENTS.md

## Repository purpose
This repo runs batch target-paper annotation.

## Inputs
- Each target paper is stored as a separate text file under `data/papers/`.
- The batch manifest is stored in `data/manifest.jsonl`.

## Tasks
For each paper, produce:
- `related_work.json`
- `gold_annotations.json`

## Global constraints
- Use only the target paper content.
- Do not use external knowledge.
- Related works must come only from the paper's own cited works / references.
- Outputs must be valid JSON.

## Output locations
Write outputs to `outputs/<paper_id>/`.
```

---

## Skill 1: `extract-related-work`

### 目标

对单篇 paper 执行以下任务：

- 输入：`paper_path`、`output_path`
- 读取 `paper_path`
- 从该 paper 的**引用 / references** 中选出最多 10 篇最重要的 related work
- 输出到 `output_path`

### 标签体系

三个标签：

- `core_comparison_work`
- `method_lineage_or_bridge`
- `tension_or_conflict_work`

### 标签定义

#### `core_comparison_work`
与 target 解决相同或高度相近的核心问题，并且是 target 在实验或性能讨论中的主要直接对比对象。

#### `method_lineage_or_bridge`
是 target 方法的明确来源、启发、继承技术线，或者是一个桥接工作：
- 方法相近但问题不同
- 问题相近但方法不同

#### `tension_or_conflict_work`
与 target 在假设、问题 framing、结论、实验设定、评价方式或经验发现上形成重要张力，并帮助解释 target paper 的 gap / motivation。

### 输出 schema

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

### skill prompt 建议

Codex 应创建 `skills/extract-related-work/SKILL.md`，建议内容如下：

```md
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
- Order related_work_items from most important to least important.

Labels:
- core_comparison_work
- method_lineage_or_bridge
- tension_or_conflict_work

Label rules:
- Assign core_comparison_work if the work solves the same or a very similar core problem and is used as a main direct comparison target in experiments or performance discussion.
- Assign method_lineage_or_bridge if the work is a clear method source, inspiration, inherited technical line, or a bridge work that connects a similar method to a different problem or a similar problem to a different method.
- Assign tension_or_conflict_work if the work creates important tension with the target paper in assumptions, framing, conclusions, evaluation setting, or empirical findings, and helps explain why the target paper's gap or motivation is needed.

A work may receive multiple labels, but must have exactly one primary_label.

Return valid JSON only with this schema:

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

---

## Skill 2: `extract-gold-annotations`

### 目标

对单篇 paper 执行以下任务：

- 输入：`paper_path`、`output_path`
- 读取 `paper_path`
- 抽取：
  - `topic`
  - `golden_gap`
  - `golden_method`
  - `gold_experiment`
- 输出到 `output_path`

### 这四项的定义

这四项都可以先输出为**连续文本**，不要求再拆成很多二级字段。重点是内容完整、抽象层级正确、紧贴论文原文。

#### `topic`
用一段连续文本描述 target paper 的核心研究主题。通常可以包括：

- 研究问题是什么
- 任务是什么
- 论文关注的对象 / 场景 / domain 是什么
- 特别的 setting 或约束是什么

#### `golden_gap`
用一段连续文本描述论文提出或隐含强调的核心 gap。通常可以包括：

- 现有方法缺什么能力
- 现有方法依赖了什么不合理假设
- 现有 evaluation / benchmark / setting 有什么不足
- 为什么这些问题足以构成这篇论文的动机

#### `golden_method`
用一段连续文本描述论文提出的方法本身。通常可以包括：

- 方法的总体思路
- 关键机制或关键模块
- 相比 prior work 的核心新意
- 方法是如何回应 golden gap 的

#### `gold_experiment`
用一段连续文本描述论文如何验证方法。通常可以包括：

- 用了哪些 benchmark / datasets / tasks
- 主要与哪些 baseline 或 prior work 比较
- 重点评估哪些性质
- 实验设计的核心验证逻辑是什么

### 输出 schema

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

### skill prompt 建议

Codex 应创建 `skills/extract-gold-annotations/SKILL.md`，建议内容如下：

```md
---
name: extract-gold-annotations
description: Extract topic, golden gap, golden method, and gold experiment from a given paper file path.
---

Input:
- paper_path: path to the target paper text file
- output_path: path to write gold_annotations.json

Read the target paper from `paper_path`.

Your task is to extract:
- topic
- golden_gap
- golden_method
- gold_experiment

Rules:
- Use only evidence from the target paper.
- Do not use external knowledge.
- Do not invent missing details.
- Prefer concise, evidence-grounded summaries.
- Write the result to `output_path`.

Definitions:
- topic: a continuous-text summary of the paper's central problem, task, domain, and setting
- golden_gap: a continuous-text summary of the main limitation, unmet need, unrealistic assumption, or evaluation problem that motivates the paper
- golden_method: a continuous-text summary of the proposed method at the right level of abstraction
- gold_experiment: a continuous-text summary of the main experimental design, datasets/benchmarks, comparisons, and evaluation focus

Return valid JSON only with this schema:

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

---

## Skill 3: `extract-paper-bundle`

### 目标

这个 skill 用于后面**批量跑 many target papers**时的主工作流。

它一次完成两个任务：

- 只读取一次 target paper
- 同时生成两个结果文件：
  - `related_work.json`
  - `gold_annotations.json`

### 输入

- `paper_id`
- `paper_path`
- `output_dir`

### 输出

- `output_dir/related_work.json`
- `output_dir/gold_annotations.json`

### skill prompt 建议

Codex 应创建 `skills/extract-paper-bundle/SKILL.md`，建议内容如下：

```md
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
- core_comparison_work
- method_lineage_or_bridge
- tension_or_conflict_work

For Task 2, extract:
- topic
- golden_gap
- golden_method
- gold_experiment

The output schemas for the two files must match the project definitions.
```

---

## 批量跑 many papers 的建议工作流

后面批量跑时，推荐流程如下：

1. 把每篇论文处理成一个文本文件，放到 `data/papers/`
2. 在 `data/manifest.jsonl` 里登记所有 paper
3. 用 Codex 或脚本逐行读取 manifest
4. 对每一篇 paper 调用 `extract-paper-bundle`
5. 将输出写到 `outputs/<paper_id>/`

### 为什么主跑 `extract-paper-bundle`

因为对于同一篇 paper：

- `extract-related-work` 和 `extract-gold-annotations` 都需要读论文正文
- 如果分两次跑，会重复读同一篇 paper
- `extract-paper-bundle` 可以让 Codex 一次读取、一次推理过程里写两个结果文件，更适合批量任务

### 为什么仍然保留两个单任务 skill

因为后面调试和 ablation 时很有用：

- 你可以单独测试 related work 抽取效果
- 你可以单独测试 gold annotation 抽取效果
- 出错时更方便定位问题

---

## 让 Codex 接下来做什么

Codex 应根据这份说明，完善项目并生成以下内容：

1. 创建或更新 `AGENTS.md`
2. 创建目录：
   - `data/papers/`
   - `outputs/`
   - `skills/extract-related-work/`
   - `skills/extract-gold-annotations/`
   - `skills/extract-paper-bundle/`
3. 写出三个 `SKILL.md`
4. 确保三个 skill 的 prompt 与本文件中的定义一致
5. 后续可再补：
   - 一个批量 runner 脚本
   - manifest 读取逻辑
   - 输出校验脚本

---

## 最终原则

这个项目的核心原则是：

- **一篇论文一个输入文件**
- **related work 只能从该论文自己的 references 中选**
- **gold annotations 只能从该论文正文中抽**
- **不使用外部知识**
- **批量场景优先使用 extract-paper-bundle**
- **单任务 skill 保留给调试与拆分评估**

