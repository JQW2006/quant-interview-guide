# 量化笔试面试指南

面向国内量化私募研究岗实习与校招的笔试面试复习资料。

在线阅读：<https://jqw2006.github.io/quant-interview-guide/>

## 内容

| 专题 | 讲解 | 题库 |
|---|---|---|
| 概率论 | [讲解](docs/probability/notes.md) | [题库](docs/probability/problems.md) |
| 统计与计量 | [讲解](docs/statistics/notes.md) | [题库](docs/statistics/problems.md) |
| 编程与算法 | [讲解](docs/programming/notes.md) | [题库](docs/programming/problems.md) |
| 数学与智力题 | [讲解](docs/math/notes.md) | [题库](docs/math/problems.md) |
| 机器学习 | [讲解](docs/machine-learning/notes.md) | [题库](docs/machine-learning/problems.md) |
| 金融常识 | [讲解](docs/finance/notes.md) | [题库](docs/finance/problems.md) |

题目标签：

- `往年·公司名`：公开面经中记录的往年笔试或面试题，题干经过整理；
- `经典·书名`：出自面试书或教材，书名缩写见各题库开头；
- `押题`：依据往年考点整理的高频题型；
- `基础` `中等` `困难`：难度。

各文件末尾列有参考来源。内容如有错误，欢迎提交 Issue 或 Pull Request。

## 本地预览

需要 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
mkdocs serve
```

浏览器打开 <http://127.0.0.1:8000>。修改 `docs/` 下的文件后页面自动刷新；修改 `mkdocs.yml` 或 `hooks/` 后需重启。

## 编写约定

文件按 GitHub 的 Markdown 写法编写，网站构建时由 `hooks/markdown.py` 转换。

- 题目标题：``### A1. 题名 `中等` `经典·绿皮书` ``，难度在前，来源标签在后；
- 答案：`<details>` 与 `<summary>答案与解析</summary>`，内部为普通 Markdown；
- 公式：`$…$`、`$$…$$`；表格内的竖线写作 `\lvert`。

构建开启了严格模式，断链、锚点失效等警告都会使构建失败：

```bash
mkdocs build
```

## 目录结构

```
docs/
├── <专题>/notes.md      讲解
├── <专题>/problems.md   题库
├── index.md             首页
└── assets/              样式、脚本、KaTeX
hooks/markdown.py        构建时的语法转换
overrides/               搜索 worker 补丁
```

`overrides/assets/javascripts/workers/search.*.min.js` 修正了 Material 搜索对中文查询的分词：原文件循环条件为 `o<t.length`，会漏掉查询的最后一个字，已改为 `o<=t.length`。`requirements.txt` 固定了 Material 版本；升级后构建会提示补丁不匹配，此时删除旧文件，从新版 `material/templates/assets/javascripts/workers/` 复制 `search.*.min.js` 并做同样修改。

## 部署

推送到 `main` 分支后，GitHub Actions（`.github/workflows/deploy.yml`）自动构建并发布到 GitHub Pages。

## 许可

本仓库内容采用 [知识共享 署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)（CC BY-NC-SA 4.0）。引用的书籍与面经题目版权归原作者所有。
