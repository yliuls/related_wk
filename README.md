# related_wk

这是一个用于处理论文相关信息提取的小仓库，主要包含两部分：

- `tag_paper/`：对目标论文做结构化抽取，生成 related work 和其他标注结果。
- `rel_paper/`：对已经生成的 `related_work.json` 做进一步汇总和统计。

## 重点目录

最重要的结果目录是：

`/Users/king/Desktop/gitcode/related_wk/tag_paper/outputs`

这个目录用于存放每篇 target paper 的处理结果。目录结构大致如下：

```text
tag_paper/outputs/
  paper_0001/
    related_work.json
    gold_annotations.json
  paper_0002/
    related_work.json
    gold_annotations.json
  ...
```

其中：

- `paper_xxxx/`：对应一篇目标论文
- `related_work.json`：这篇论文抽取出的 related work 结果
- `gold_annotations.json`：这篇论文抽取出的其他关键信息标注结果

## 仓库作用

这个仓库主要用于：

1. 批量处理目标论文
2. 为每篇论文生成结构化 JSON 结果
3. 在已有结果基础上继续做 related paper 的索引和统计

## 相关目录

- `tag_paper/data/`：输入数据和 manifest
- `tag_paper/outputs/`：核心输出结果目录
- `rel_paper/`：对 `related_work.json` 结果做聚合统计的脚本和说明
