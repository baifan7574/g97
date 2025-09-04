#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ads_apply_all_inline.py
- 兼容老脚本字段：home/inner 下的 top_banner, bottom_banner, popup
- 新增支持：inner/home 下的 inline_banner（正文中部广告，非悬浮）
- 如果配置里没有 floating，则自动清理历史悬浮条 .nb-bottombar-wrap / nb-has-bottom
- 首次运行前建议备份站点文件
"""

import re
import json
import pathlib

ROOT = pathlib.Path(".")
CONF = ROOT / "ads_mapping.json"

# —— 注入标记，避免重复 —— #
MARKS = {
    "top":    ("<!-- NB:AD-TOP START -->",    "<!-- NB:AD-TOP END -->"),
    "bottom": ("<!-- NB:AD-BOTTOM START -->", "<!-- NB:AD-BOTTOM END -->"),
    "inline": ("<!-- NB:AD-INLINE START -->", "<!-- NB:AD-INLINE END -->"),
    "popup":  ("<!-- NB:AD-POPUP START -->",  "<!-- NB:AD-POPUP END -->"),
}

BODY_OPEN_RE  = re.compile(r"<body\b[^>]*>", re.I)
BODY_CLOSE_RE = re.compile(r"</body\s*>", re.I)

# 常见“正文锚点”用于插入 inline（择一命中）
INLINE_ANCHORS = [
    # 显式占位（若你在模板里预留了这个注释，会优先命中）
    re.compile(r"<!--\s*NB:INLINE-ANCHOR\s*-->", re.I),
    # 首个 <h2> 后
    re.compile(r"</h2\s*>", re.I),
    # 首个段落后
    re.compile(r"</p\s*>", re.I),
    # 首个图片后
    re.compile(r"</img\s*>", re.I),
    # main 结束前（兜底）
    re.compile(r"</main\s*>", re.I),
]

# 清理历史悬浮条
def clean_legacy_floating(html: str) -> str:
    s = html
    # 去掉底部悬浮包装
    s = re.sub(r'<div[^>]*class="[^"]*\bnb-bottombar-wrap\b[^"]*"[^>]*>[\s\S]*?</div>\s*', "", s, flags=re.I)
    # 去掉 body 上的 nb-has-bottom
    s = re.sub(r'(<body\b[^>]*class="[^"]*)\bnb-has-bottom\b\s*', r"\1", s, flags=re.I)
    # 如果 body 没有 class 属性，也容错处理（无事发生）
    return s

def already_has(mark_start: str, html: str) -> bool:
    return mark_start in html

def inject_after_body_open(html: str, block: str, mark_key: str) -> str:
    m = BODY_OPEN_RE.search(html)
    if not m: return html
    start, end = m.span()
    s, e = MARKS[mark_key]
    snippet = f"\n{s}\n{block}\n{e}\n"
    return html[:end] + snippet + html[end:]

def inject_before_body_close(html: str, block: str, mark_key: str) -> str:
    m = BODY_CLOSE_RE.search(html)
    if not m: return html + block
    s, e = MARKS[mark_key]
    snippet = f"\n{s}\n{block}\n{e}\n"
    pos = m.start()
    return html[:pos] + snippet + html[pos:]

def inject_inline(html: str, block: str) -> str:
    s_mark, e_mark = MARKS["inline"]
    snippet = f"\n{s_mark}\n{block}\n{e_mark}\n"
    # 1) 优先命中显式占位
    if "<!-- NB:AD-INLINE START -->" in html:
        # 已有则不重复
        return html
    for rex in INLINE_ANCHORS:
        m = rex.search(html)
        if m:
            pos = m.end()
            return html[:pos] + snippet + html[pos:]
    # 2) 如果都没命中，就放在 </body> 前（兜底，仍是非 fixed）
    return inject_before_body_close(html, block, "inline")

def wrap_popup_with_cooldown(code: str, hours: int = 6) -> str:
    """在不改联盟代码的前提下，加一层本地频控（localStorage）。"""
    wrapper = f"""
<script>
(function(){{
  try {{
    var key='nb_pop_t';
    var hours={hours};
    var last=localStorage.getItem(key);
    var ok=!last || (Date.now()-(+last))/36e5>=hours;
    if(!ok) return;
    // 触发联盟弹窗代码：
  }} catch(e) {{}}
}})();
</script>
{code}
<script>try{{localStorage.setItem('nb_pop_t', String(Date.now()));}}catch(e){{}}</script>
"""
    return wrapper.strip()

def pick_role(path: pathlib.Path) -> str:
    """简单区分首页/内页：文件名含 index/home 视为首页，其它为内页。"""
    name = path.name.lower()
    if name in ("index.html", "home.html") or name.endswith("/index.html"):
        return "home"
    return "inner"

def main():
    if not CONF.exists():
        print("ads_mapping.json not found.")
        return
    cfg = json.loads(CONF.read_text(encoding="utf-8"))

    enable_home  = cfg.get("global", {}).get("enable_on_home", True)
    enable_inner = cfg.get("global", {}).get("enable_on_inner", True)

    # 是否存在 floating 配置（如果没有，等会顺手清理历史悬浮）
    has_floating_in_conf = any("floating" in cfg.get(k, {}) for k in ("home", "inner"))

    files = list(ROOT.rglob("*.html"))
    for f in files:
        role = pick_role(f)
        if role == "home" and not enable_home:
            continue
        if role == "inner" and not enable_inner:
            continue

        html = f.read_text(encoding="utf-8", errors="ignore")
        original = html

        section = cfg.get(role, {})

        # 1) 顶部
        tb_list = section.get("top_banner", [])
        if tb_list and not already_has(MARKS["top"][0], html):
            block = "\n".join(tb_list)
            html = inject_after_body_open(html, block, "top")

        # 2) 中部（新的 inline_banner，非 fixed）
        inl_list = section.get("inline_banner", [])
        if inl_list and not already_has(MARKS["inline"][0], html):
            block = "\n".join(inl_list)
            html = inject_inline(html, block)

        # 3) 底部
        bb_list = section.get("bottom_banner", [])
        if bb_list and not already_has(MARKS["bottom"][0], html):
            block = "\n".join(bb_list)
            html = inject_before_body_close(html, block, "bottom")

        # 4) 弹窗（加 6 小时冷却包装）
        pp_list = section.get("popup", [])
        if pp_list and not already_has(MARKS["popup"][0], html):
            raw_code = "\n".join(pp_list)
            wrapped = wrap_popup_with_cooldown(raw_code, hours=1)
            html = inject_before_body_close(html, wrapped, "popup")

        # 5) 如果配置里没有 floating，就顺手清理历史悬浮（一次性清理/幂等）
        if not has_floating_in_conf:
            html = clean_legacy_floating(html)

        if html != original:
            f.write_text(html, encoding="utf-8")
            print("updated:", f)

    print("done.")

if __name__ == "__main__":
    main()
