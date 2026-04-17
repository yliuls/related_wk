# rel_paper

用于汇总一批 `related_work.json`，生成：

- `paper_idx -> related papers`
- `related paper -> citing paper_idx`

输入目录需要满足类似 `tag_paper/outputs` 的结构：

```text
some_input_dir/
  paper_0001/
    related_work.json
  paper_0002/
    related_work.json
  ...
```

## 使用命令

在仓库根目录运行：

```bash
python3 rel_paper/build_related_paper_index.py tag_paper/outputs \
  --paper-to-related-output rel_paper/paper_to_related_papers.json \
  --related-to-paper-output rel_paper/related_paper_to_citing_papers.json
```

如果你的输入目录不是 `tag_paper/outputs`，把第一个参数替换成你的目录即可。

例如：

```bash
python3 rel_paper/build_related_paper_index.py /path/to/your/outputs \
  --paper-to-related-output rel_paper/paper_to_related_papers.json \
  --related-to-paper-output rel_paper/related_paper_to_citing_papers.json
```

## 输出文件

### 1. `paper_to_related_papers.json`

记录每个 `paper_idx` 对应的：

- `target_paper_title`
- `related_papers`
- `related_paper_count`

同时包含整体统计信息：

- `processed_subdirectories`
- `papers_with_related_work`
- `missing_related_work_files`

### 2. `related_paper_to_citing_papers.json`

记录每篇 `related paper` 对应的：

- `citing_papers`
- `citation_count`

用于查看某篇 related paper 被哪些 target paper 引用。

## 说明

- 脚本会扫描输入目录下的每个一级子目录。
- 每个子目录默认读取 `related_work.json`。
- 同一个 `paper_idx` 内部重复出现的 `paper_title` 会自动去重。
- 缺失 `related_work.json` 的目录不会报错退出，而是记录到输出统计中。
