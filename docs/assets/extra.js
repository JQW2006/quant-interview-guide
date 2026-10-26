/* 公式渲染、题库筛选、答案折叠、检查清单持久化 */

function renderMath(root) {
  renderMathInElement(root, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
    strict: "ignore",
  });
}

/* 行内公式宽于容器时（窄屏），改为可横向滚动的块，避免整页横向滚动 */
function fitWideMath(root) {
  root.querySelectorAll("span.arithmatex").forEach((span) => {
    span.classList.remove("math-wide");
    const box = span.parentElement;
    if (box && box.clientWidth && span.getBoundingClientRect().width > box.clientWidth) {
      span.classList.add("math-wide");
    }
  });
}

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => fitWideMath(document), 200);
});
document.addEventListener(
  "toggle",
  (e) => {
    if (e.target.open) fitWideMath(e.target);
  },
  true,
);

function categoryOf(tag) {
  const head = tag.split("·")[0];
  if (head.startsWith("往年")) return "往年";
  if (head.startsWith("经典")) return "经典";
  if (head.startsWith("押题")) return "押题";
  return null;
}

/* 把每道题（h3 到下一个 h2/h3/hr 之间）包进 section.q */
function wrapQuestions(article) {
  const questions = [];
  for (const h3 of article.querySelectorAll(":scope > h3")) {
    const level = h3.querySelector("code.tag--level");
    if (!level) continue;
    const cats = new Set();
    h3.querySelectorAll("code").forEach((c) => {
      const cat = categoryOf(c.textContent);
      if (cat) cats.add(cat);
    });
    const section = document.createElement("section");
    section.className = "q";
    section.dataset.level = level.textContent;
    section.dataset.cats = [...cats].join(" ");
    h3.before(section);
    let node = h3;
    while (node && !(node !== h3 && /^(H1|H2|H3|HR)$/.test(node.tagName))) {
      const next = node.nextElementSibling;
      section.append(node);
      node = next;
    }
    questions.push(section);
  }
  return questions;
}

function buildToolbar(article, questions) {
  const bar = document.createElement("div");
  bar.className = "q-toolbar";
  bar.innerHTML = `
    <div class="q-row">
      <span class="q-label">难度</span>
      <button type="button" data-f="level" data-v="基础">基础</button>
      <button type="button" data-f="level" data-v="中等">中等</button>
      <button type="button" data-f="level" data-v="困难">困难</button>
    </div>
    <div class="q-row">
      <span class="q-label">标签</span>
      <button type="button" class="chip--past" data-f="cats" data-v="往年">往年</button>
      <button type="button" class="chip--classic" data-f="cats" data-v="经典">经典</button>
      <button type="button" class="chip--guess" data-f="cats" data-v="押题">押题</button>
    </div>
    <div class="q-row q-actions">
      <button type="button" data-a="open">全部展开</button>
      <button type="button" data-a="close">全部折叠</button>
      <button type="button" data-a="reset">清除筛选</button>
      <span class="q-status"></span>
    </div>`;

  const status = bar.querySelector(".q-status");
  const selected = { level: new Set(), cats: new Set() };

  function apply() {
    let shown = 0;
    for (const q of questions) {
      const okLevel = !selected.level.size || selected.level.has(q.dataset.level);
      const cats = q.dataset.cats.split(" ").filter(Boolean);
      const okCats = !selected.cats.size || cats.some((c) => selected.cats.has(c));
      const visible = okLevel && okCats;
      q.hidden = !visible;
      if (visible) shown++;
    }
    // 章节内无可见题目时，隐藏章节标题与章节说明
    for (const h2 of article.querySelectorAll(":scope > h2")) {
      const block = [];
      let any = false, hasQ = false;
      for (let n = h2.nextElementSibling; n && n.tagName !== "H2"; n = n.nextElementSibling) {
        if (n.matches("section.q")) {
          hasQ = true;
          if (!n.hidden) any = true;
        } else block.push(n);
      }
      if (hasQ) [h2, ...block].forEach((n) => (n.hidden = !any));
    }
    status.textContent = `显示 ${shown} / ${questions.length} 题`;
  }

  bar.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    if (btn.dataset.f) {
      const set = selected[btn.dataset.f];
      set.has(btn.dataset.v) ? set.delete(btn.dataset.v) : set.add(btn.dataset.v);
      btn.classList.toggle("is-active");
      apply();
    } else if (btn.dataset.a === "reset") {
      selected.level.clear();
      selected.cats.clear();
      bar.querySelectorAll(".is-active").forEach((b) => b.classList.remove("is-active"));
      apply();
    } else {
      const open = btn.dataset.a === "open";
      article.querySelectorAll("details.answer").forEach((d) => (d.open = open));
    }
  });

  // 本页没有的标签不显示
  bar.querySelectorAll('[data-f="cats"]').forEach((b) => {
    b.hidden = !questions.some((q) => q.dataset.cats.split(" ").includes(b.dataset.v));
  });

  // 放在标题和开头说明之后
  const h1 = article.querySelector(":scope > h1");
  let anchor = h1;
  while (anchor && anchor.nextElementSibling && /^(P|DIV|TABLE|BLOCKQUOTE)$/.test(anchor.nextElementSibling.tagName)) {
    anchor = anchor.nextElementSibling;
  }
  (anchor || article.firstElementChild).after(bar);
  apply();
}

/* 检查清单：勾选状态存 localStorage，键为页面路径 + 条目文字 */
function persistChecklists(article) {
  const boxes = article.querySelectorAll(".task-list-item input[type=checkbox]");
  if (!boxes.length) return;
  const key = "checklist:" + decodeURIComponent(location.pathname);
  let state = {};
  try {
    state = JSON.parse(localStorage.getItem(key) || "{}");
  } catch (e) {}
  boxes.forEach((box) => {
    const id = box.closest("li").textContent.trim();
    if (id in state) box.checked = state[id];
    box.addEventListener("change", () => {
      state[id] = box.checked;
      try {
        localStorage.setItem(key, JSON.stringify(state));
      } catch (e) {}
    });
  });
}

document$.subscribe(({ body }) => {
  renderMath(body);
  const article = body.querySelector("article.md-content__inner");
  if (!article) return;
  if (/\/problems\/?$/.test(location.pathname)) {
    const questions = wrapQuestions(article);
    if (questions.length) buildToolbar(article, questions);
  }
  persistChecklists(article);
  fitWideMath(article);

  // 公式渲染改变了页面高度，重新定位到锚点
  const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (target) target.scrollIntoView();
});
