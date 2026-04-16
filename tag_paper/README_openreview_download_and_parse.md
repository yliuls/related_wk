# OpenReview PDF Download and SciPDF Parse

这个目录下的脚本 [download_and_parse_openreview.py](/data3/yaofu/related_wk/tag_paper/download_and_parse_openreview.py) 用来：

- 从输入 JSON 中读取论文列表
- 根据每条论文的 `openreview_url` / `openreview_id` 下载 PDF
- 用 `title` 作为 PDF 文件名
- 调用 SciPDF Parser 解析 PDF
- 只保留正文到 `references` 为止，忽略 appendix / supplementary 之后的正文部分
- 将 PDF 和解析后的 JSON 分别保存到两个目录

## 1. 输入 JSON 格式

示例文件：

- [candidate_papers_os_ar.json](/data3/yaofu/related_wk/tag_paper/candidate_papers_os_ar.json)

脚本要求输入 JSON 是一个 list，每个元素至少包含：

- `title`
- `openreview_url`
- `openreview_id`

## 2. 依赖

脚本依赖：

- Python 3
- `openreview-py`
- `requests`
- `scipdf`
- 本地可访问的 GROBID 服务

其中 SciPDF Parser 实际解析 PDF 时会调用 GROBID，默认地址是：

```bash
http://localhost:8070
```

如果你的 GROBID 不在这个地址，需要通过 `--grobid-url` 指定。

## 3. OpenReview 账号配置

推荐使用环境变量，不要把密码直接写在命令行里。

```bash
export OPENREVIEW_USERNAME='your_email@example.com'
export OPENREVIEW_PASSWORD='your_password'
```

脚本会优先读取：

- `OPENREVIEW_USERNAME`
- `OPENREVIEW_PASSWORD`

如果你显式传了命令行参数：

- `--openreview-username`
- `--openreview-password`

那么命令行参数优先。

## 4. 基本用法

```bash
python /data3/yaofu/related_wk/tag_paper/download_and_parse_openreview.py \
  --input-json /data3/yaofu/related_wk/tag_paper/candidate_papers_os_ar.json \
  --pdf-dir /data3/yaofu/related_wk/tag_paper/pdfs \
  --parsed-json-dir /data3/yaofu/related_wk/tag_paper/parsed_json \
  --start 1 \
  --end 10 \
  --download-method openreview-api
```

这个命令会：

- 处理输入 JSON 中第 1 到第 10 篇论文
- 将 PDF 保存到 `pdfs/`
- 将解析后的 JSON 保存到 `parsed_json/`
- 优先通过 OpenReview 官方客户端/API 下载 PDF

## 5. 常用参数

- `--input-json`
  输入论文列表 JSON 文件

- `--pdf-dir`
  PDF 输出目录

- `--parsed-json-dir`
  解析 JSON 输出目录

- `--start`
  起始论文编号，1-based，包含本项

- `--end`
  结束论文编号，1-based，包含本项

- `--skip-existing`
  如果 PDF 和解析 JSON 都已存在，则跳过

- `--grobid-url`
  GROBID 服务地址，默认 `http://localhost:8070`

- `--download-method`
  可选值：
  - `openreview-api`：只走 OpenReview 官方客户端
  - `direct`：只走 PDF 直链
  - `auto`：先尝试官方客户端，失败后回退直链

- `--openreview-baseurl`
  OpenReview API 地址，默认：

```bash
https://api2.openreview.net
```

## 6. 只跑一小段

例如只处理第 11 到第 20 篇：

```bash
python /data3/yaofu/related_wk/tag_paper/download_and_parse_openreview.py \
  --input-json /data3/yaofu/related_wk/tag_paper/candidate_papers_os_ar.json \
  --pdf-dir /data3/yaofu/related_wk/tag_paper/pdfs \
  --parsed-json-dir /data3/yaofu/related_wk/tag_paper/parsed_json \
  --start 11 \
  --end 20 \
  --download-method openreview-api \
  --skip-existing
```

## 7. 输出结果

输出有两类：

- PDF 文件：
  保存在你指定的 `--pdf-dir`

- 解析后的 JSON：
  保存在你指定的 `--parsed-json-dir`

JSON 中主要包含：

- `title`
- `authors`
- `pub_date`
- `abstract`
- `sections`
- `references`
- `figures`
- `formulas`
- `doi`
- `source_paper`

其中：

- `sections` 会在遇到 appendix / supplementary 标题时截断
- `references` 会保留
- `source_paper` 会附带原输入 JSON 里的论文元信息

## 8. 运行示例

先设置账号：

```bash
export OPENREVIEW_USERNAME='your_email@example.com'
export OPENREVIEW_PASSWORD='your_password'
```

再运行：

```bash
python /data3/yaofu/related_wk/tag_paper/download_and_parse_openreview.py \
  --input-json /data3/yaofu/related_wk/tag_paper/candidate_papers_os_ar.json \
  --pdf-dir /data3/yaofu/related_wk/tag_paper/pdfs \
  --parsed-json-dir /data3/yaofu/related_wk/tag_paper/parsed_json \
  --start 1 \
  --end 10 \
  --download-method openreview-api \
  --skip-existing
```

## 9. 常见问题

### 9.1 OpenReview 下载返回 403

通常表示匿名访问权限不够。解决方法：

- 配置 `OPENREVIEW_USERNAME`
- 配置 `OPENREVIEW_PASSWORD`
- 使用 `--download-method openreview-api`

### 9.2 SciPDF 解析失败

SciPDF 依赖 GROBID。请先确认：

- GROBID 服务已经启动
- `--grobid-url` 配置正确
- 本机可以访问该地址

例如：

```bash
curl http://localhost:8070
```

### 9.3 已经下载过，不想重复跑

加上：

```bash
--skip-existing
```

## 10. 查看帮助

```bash
python /data3/yaofu/related_wk/tag_paper/download_and_parse_openreview.py --help
```
