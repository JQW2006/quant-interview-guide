"""构建钩子：把 GitHub 风格的 Markdown 转换为 Python-Markdown 可识别的形式，并为题目标签着色。

源文件保持 GitHub 可读的写法（<details> 答案、紧贴段落的列表与公式），网站构建时在此转换。
"""

import logging
import re
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.markdown")

ROOT = Path(__file__).resolve().parent.parent

DETAILS_RE = re.compile(r"^<details>\s*(.*?)\s*$")
SUMMARY_RE = re.compile(r"^<summary>(.*)</summary>\s*$")
QUOTE_RE = re.compile(r"^(?: {0,3}> ?)*")
LIST_ITEM_RE = re.compile(r"^( *)([-*+]|\d+[.)])(?: +|$)")
FENCE_RE = re.compile(r"^(\s*(?:>\s*)*)(`{3,}|~{3,})(.*)$")
MATH_LINE_RE = re.compile(r"^(\s*(?:>\s*)*)(\$\$.*)$")

H3_RE = re.compile(r"<h3 .*?</h3>", re.S)
TAG_RE = re.compile(r"<code>([^<]*)</code>")
LEVELS = ("基础", "中等", "困难")
LEVEL_RE = re.compile(rf" ({'|'.join(LEVELS)})(?= |$)")
TAG_KINDS = {"past": "往年", "classic": "经典", "guess": "押题"}


def on_config(config):
    # overrides/ 中的搜索 worker 是对 Material 自带文件的单字符补丁（中文查询漏掉最后一个字），
    # 文件名含哈希，升级 Material 后需按 README 重新生成
    import material

    worker = "assets/javascripts/workers"
    theme = Path(material.__file__).parent / "templates" / worker
    patched = {p.name for p in (ROOT / "overrides" / worker).glob("search.*.min.js")}
    if not patched & {p.name for p in theme.glob("search.*.min.js")}:
        log.warning("搜索 worker 补丁与当前 Material 版本不匹配，请按 README 重新生成")
    return config


def on_serve(server, config, builder):
    server.watch(str(ROOT / "hooks"))
    return server


def on_page_markdown(markdown, page, config, files):
    markdown = _isolate_display_math(markdown)
    markdown = _normalize_lists(markdown)
    return _convert_details(markdown)


def _normalize_lists(markdown):
    """把 GitHub 风格的列表写法转换为 Python-Markdown 可识别的形式。

    1. 列表紧跟段落文字时（GitHub 允许），在前面补空行；
    2. 列表内 1–3 个空格的缩进（嵌套列表、续行、代码块）补齐到 4 个空格。
    """
    out = []
    fence = None  # (标记, 缩进增量)
    in_list = loose = False
    prev_prefix, prev_rest = None, ""
    for line in markdown.split("\n"):
        q = QUOTE_RE.match(line)
        prefix, rest = q.group(0), line[q.end():]
        if prefix != prev_prefix:
            in_list = False

        if fence is not None:
            mark, delta = fence
            if rest.strip().startswith(mark) and not rest.strip().strip(mark[0]):
                fence = None
            out.append(prefix + " " * delta + rest if rest.strip() else line)
            prev_prefix, prev_rest = prefix, rest
            continue

        indent = len(rest) - len(rest.lstrip(" "))
        item = LIST_ITEM_RE.match(rest)
        blank = not rest.strip()

        if item and not in_list:
            ordered = item.group(2)[0].isdigit()
            if (prev_rest.strip() and prev_prefix == prefix and not prev_rest.lstrip().startswith("#")
                    and (not ordered or int(item.group(2)[:-1]) == 1)):
                out.append(prefix.rstrip())
            in_list, loose = True, False
        elif item and loose and prev_rest.strip() and not LIST_ITEM_RE.match(prev_rest):
            # 上一项内已有空行分段（如独立公式），续行段落后的新项需空行分隔
            out.append(prefix.rstrip())
            loose = False
        elif item:
            loose = False
        elif in_list and blank:
            loose = True
        elif in_list and not blank and indent == 0 and not item and not prev_rest.strip():
            in_list = False
        elif not blank and rest.lstrip().startswith("#"):
            in_list = False

        delta = 0
        if in_list and 0 < indent < 4:
            delta = 4 - indent
            rest = " " * delta + rest
        f = FENCE_RE.match(rest)
        if f:
            fence = (f.group(2), delta)
        out.append(prefix + rest)
        prev_prefix, prev_rest = prefix, rest
    return "\n".join(out)


def _isolate_display_math(markdown):
    """独占一行的 $$…$$（或多行 $$ 块）前后补空行。

    GitHub 允许 $$ 紧贴段落文字，arithmatex 要求公式自成一块。
    代码块内不处理；保留引用前缀 `>` 与列表缩进。
    """
    out, fence, in_math = [], None, False
    for line in markdown.split("\n"):
        m = FENCE_RE.match(line)
        if m and not in_math:
            mark = m.group(2)
            if fence is None:
                fence = mark
            elif mark.startswith(fence[0]) and len(mark) >= len(fence) and not m.group(3).strip():
                fence = None
            out.append(line)
            continue
        if fence is not None:
            out.append(line)
            continue
        if in_math:
            out.append(line)
            if line.rstrip().endswith("$$"):
                in_math = False
                out.append(blank)
            continue
        p = MATH_LINE_RE.match(line)
        if not p:
            out.append(line)
            continue
        prefix, body = p.group(1), p.group(2).rstrip()
        blank = prefix.rstrip() if ">" in prefix else ""
        single = len(body) > 4 and body.endswith("$$") and body.count("$$") == 2
        if not single and body.count("$$") != 1:
            out.append(line)  # 如 $$a$$ 后接正文，按行内处理
            continue
        if out and out[-1].strip(" >"):
            out.append(blank)
        out.append(line)
        if single:
            out.append(blank)
        else:
            in_math = True
    return "\n".join(out)


def _convert_details(markdown):
    """<details><summary>T</summary> … </details> 转为 pymdownx.details 的 ??? 语法。

    原样保留 HTML 时，md_in_html 会把公式中的 `$x<y$` 当作标签起点，吞掉后文；
    转成缩进块后，内部完全按 Markdown 解析。
    """
    out, body, title = [], None, None
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if body is None and DETAILS_RE.match(line):
            rest = DETAILS_RE.match(line).group(1)
            if rest:
                m, skip = SUMMARY_RE.match(rest), 1
            else:
                m, skip = SUMMARY_RE.match(lines[i + 1]) if i + 1 < len(lines) else None, 2
            if m:
                title, body = m.group(1).strip(), []
                i += skip
                continue
        if body is not None and line.strip() == "</details>":
            while body and not body[-1].strip():
                body.pop()
            while body and not body[0].strip():
                body.pop(0)
            out.append(f'??? answer "{title}"')
            out.append("")
            out.extend(("    " + b) if b.strip() else "" for b in body)
            out.append("")
            body = None
        elif body is not None:
            body.append(line)
        else:
            out.append(line)
        i += 1
    if body is not None:
        raise ValueError("未闭合的 <details>")
    return "\n".join(out)


def on_page_content(html, page, config, files):
    # 右侧目录只保留题号与题名，去掉难度与标签
    def trim(items):
        for item in items:
            cut = list(LEVEL_RE.finditer(item.title))
            if cut:
                item.title = item.title[:cut[-1].start()]
            trim(item.children)

    trim(page.toc)

    # 题目标题中的难度与来源标签
    def tag(m):
        text = m.group(1)
        if text in LEVELS:
            return f'<code class="tag tag--level">{text}</code>'
        kind = next((k for k, v in TAG_KINDS.items() if text.startswith(v)), None)
        return f'<code class="tag tag--{kind}">{text}</code>' if kind else m.group(0)

    return H3_RE.sub(lambda h: TAG_RE.sub(tag, h.group(0)), html)
