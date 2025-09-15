# -*- coding: utf-8 -*-
"""
patch_nb_variants.py
为图片详情页注入“黑框模块”，多样式+多配色，稳定随机，内链去重。
用法：
  python tools/patch_nb_variants.py --site-root D:\sites\g99 --modules-per-page 2

建议在 site_enhance_all.py 之后、sitemap_fix.py 之前运行。
"""

import re, argparse, hashlib, random
from pathlib import Path
from bs4 import BeautifulSoup  # pip install beautifulsoup4

PALETTES = [
  # (类名, 背景, 边框, 文字, 次级文字, 标签边, 标签背景, 标签文字, 标题)
  ("dark",     "#111", "#2a2a2a", "#e5e5e5", "#bfbfbf", "#3a3a3a", "rgba(255,255,255,0.04)", "#e5e5e5", "#fff"),
  ("slate",    "#0f172a", "#1e293b", "#e2e8f0", "#94a3b8", "#334155", "rgba(148,163,184,0.08)", "#e2e8f0", "#f8fafc"),
  ("midnight", "#0b1020", "#1a2240", "#dbeafe", "#93c5fd", "#29325a", "rgba(59,130,246,0.10)", "#dbeafe", "#eef2ff"),
  ("indigo",   "#111827", "#312e81", "#e0e7ff", "#c7d2fe", "#4338ca", "rgba(99,102,241,0.10)", "#e0e7ff", "#ffffff"),
  ("emerald",  "#052e2b", "#064e3b", "#d1fae5", "#a7f3d0", "#065f46", "rgba(16,185,129,0.10)", "#d1fae5", "#ecfdf5"),
  ("rose",     "#2b0b12", "#4c0519", "#ffe4e6", "#fecdd3", "#881337", "rgba(244,63,94,0.10)", "#ffe4e6", "#fff1f2"),
]

CSS_BASE = """
/* === NB Black Box Variants (multi-theme) === */
.nb-box{border-radius:14px;padding:14px;margin:18px 0}
.nb-box h3{margin:0 0 10px 0;font-size:18px}
.nb-muted{opacity:.85}
.nb-chip{display:inline-block;margin:6px 8px 0 0;padding:6px 10px;border-radius:999px}
.nb-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.nb-grid a{display:block;border-radius:12px;overflow:hidden}
.nb-grid img{width:100%;height:120px;object-fit:cover;display:block}
.nb-carousel{display:flex;overflow-x:auto;gap:10px;padding-bottom:6px;scroll-snap-type:x mandatory}
.nb-carousel a{flex:0 0 180px;scroll-snap-align:start;border-radius:12px;overflow:hidden}
.nb-carousel img{width:100%;height:120px;object-fit:cover;display:block}
.nb-list{display:grid;grid-template-columns:1fr;gap:10px}
.nb-list a{display:flex;gap:10px;align-items:center;border-radius:12px;overflow:hidden;padding:8px}
.nb-list img{width:84px;height:84px;object-fit:cover;border-radius:10px}
.nb-right{display:grid;grid-template-columns:1fr 220px;gap:12px;align-items:start}
@media (max-width: 780px){.nb-right{grid-template-columns:1fr}}
"""

def css_theme_block():
    parts = [CSS_BASE]
    for name, bg, br, fg, fg2, chip_bd, chip_bg, chip_fg, hcol in PALETTES:
        parts.append(f"""
/* theme: {name} */
.nb-{name}{{background:{bg};border:1px solid {br};color:{fg}}}
.nb-{name} h3{{color:{hcol}}}
.nb-{name} .nb-muted{{color:{fg2}}}
.nb-{name} .nb-chip{{border:1px solid {chip_bd};background:{chip_bg};color:{chip_fg}}}
.nb-{name} a{{color:{chip_fg};text-decoration:none}}
.nb-{name} a:hover{{opacity:.9}}
.nb-{name} .nb-grid a, .nb-{name} .nb-carousel a{{border:1px solid {br}}}
.nb-{name} .nb-list a{{border:1px solid {br};background:rgba(255,255,255,0.02)}}
""")
    return "\n".join(parts)

def is_detail_page(name:str)->bool:
    # 匹配你的详情页命名：20250817_090303_01.html
    return bool(re.match(r"\d{8}_\d{6}_\d+\.html$", name))

def md5_int(s:str)->int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

def stable_pick(seq, seed:str):
    if not seq: return None
    idx = md5_int(seed) % len(seq)
    return seq[idx]

def ensure_css(soup:BeautifulSoup):
    head = soup.head or soup.new_tag("head")
    if not soup.head: soup.html.insert(0, head)
    # 已注入则跳过
    for st in head.find_all("style"):
        if st.string and "NB Black Box Variants" in st.string:
            return
    style = soup.new_tag("style")
    style.string = css_theme_block()
    head.append(style)

def collect_links(site_root:Path, cur:Path, need:int=12):
    rels = []
    # 1) 同目录优先
    same = [p for p in cur.parent.glob("*.html") if p != cur]
    random.shuffle(same)
    for p in same[:need*2]:
        rels.append("/" + p.relative_to(site_root).as_posix())

    # 2) 其它目录混入
    if len(rels) < need:
        others = [p for p in site_root.rglob("*.html") if p.parent != cur.parent]
        random.shuffle(others)
        for p in others[:need*4]:
            rels.append("/" + p.relative_to(site_root).as_posix())

    # 去重截断
    out, seen = [], set()
    for h in rels:
        if h in seen: continue
        seen.add(h)
        out.append(h)
        if len(out) >= need: break
    return out

def thumb_src(href:str):
    # 尝试把 .html 替换成 .jpg 作为缩略图路径（按你的静态规则）
    return href.replace(".html", ".jpg")

def render_module_html(variant:str, theme:str, links:list, seed:str):
    # 为了稳定随机，基于 seed 决定标题文案
    titles = {
        "tags":   ["Top Collections", "Explore more", "You may also like"],
        "grid":   ["Gallery picks", "Curated sets", "Fresh looks"],
        "carousel":["Discover more", "Trending now", "Also see"],
        "list":   ["Related stories", "Browse nearby", "Similar entries"],
        "right":  ["Handpicked", "Editor’s picks", "More in this set"],
    }
    title = stable_pick(titles.get(variant, ["Discover"]), seed)

    if variant == "tags":
        chips = "".join([f'<a class="nb-chip" href="{h}">Explore</a>' for h in links])
        inner = f'<h3>{title}</h3><div>{chips}</div>'

    elif variant == "grid":
        grid = "".join([f'<a href="{h}"><img loading="lazy" src="{thumb_src(h)}" alt="related"></a>' for h in links])
        inner = f'<h3>{title}</h3><div class="nb-grid">{grid}</div>'

    elif variant == "carousel":
        items = "".join([f'<a href="{h}"><img loading="lazy" src="{thumb_src(h)}" alt="see also"></a>' for h in links])
        inner = f'<h3>{title}</h3><div class="nb-carousel">{items}</div>'

    elif variant == "list":
        # 列表：左图右文
        items = "".join([f'<a href="{h}"><img loading="lazy" src="{thumb_src(h)}" alt=""><span class="nb-muted">{h.rsplit("/",1)[-1].replace(".html","").replace("_"," ")}</span></a>' for h in links[:8]])
        inner = f'<h3>{title}</h3><div class="nb-list">{items}</div>'

    else:  # right 布局：右侧小图，左侧标签
        chips = "".join([f'<a class="nb-chip" href="{h}">Open</a>' for h in links[:8]])
        thumbs = "".join([f'<a href="{h}"><img loading="lazy" src="{thumb_src(h)}" alt=""></a>' for h in links[8:16]])
        inner = f'<h3>{title}</h3><div class="nb-right"><div>{chips}</div><div class="nb-grid">{thumbs}</div></div>'

    return f'<section class="nb-box nb-{theme}">{inner}</section>'

def inject_modules(site_root:Path, html_path:Path, modules_per_page:int=2, salt:str=""):
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    ensure_css(soup)
    body = soup.body or soup.new_tag("body")
    if not soup.body: soup.append(body)

    # 已有 nb-box 则不再插，保持幂等
    if soup.select_one(".nb-box"):
        return False

    seed_base = "/" + html_path.relative_to(site_root).as_posix() + salt
    # 稳定决定模块数（1..modules_per_page）
    maxn = max(1, modules_per_page)
    count = (md5_int(seed_base) % maxn) + 1

    # 准备链接池
    all_links = collect_links(site_root, html_path, need=20)

    VARIANTS = ["tags", "grid", "carousel", "list", "right"]
    changed = False
    anchor = None
    # 尽量放在正文 AUTO_DESC 前
    for c in soup.find_all(string=lambda x: isinstance(x, str) and "AUTO_DESC_START" in x):
        anchor = c
        break

    for i in range(count):
        v_seed = f"{seed_base}#v{i}"
        t_seed = f"{seed_base}#t{i}"
        variant = stable_pick(VARIANTS, v_seed)
        theme   = PALETTES[md5_int(t_seed) % len(PALETTES)][0]

        # 将链接集合打乱且去重（不同模块拿到的子集不同）
        random.Random(md5_int(v_seed)).shuffle(all_links)
        use_links = all_links[: (12 if variant in ("grid","carousel","right") else 10)]

        block_html = render_module_html(variant, theme, use_links, v_seed)
        block = BeautifulSoup(block_html, "html.parser")

        if anchor and anchor.parent:
            anchor.parent.insert_before(block)
        else:
            body.append(block)
        changed = True

    if changed:
        html_path.write_text(str(soup), encoding="utf-8")
    return changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-root", required=True, help="站点根目录")
    ap.add_argument("--modules-per-page", type=int, default=2, help="每页最多插入几个模块（稳定随机 1..N）")
    ap.add_argument("--salt", default="", help="可选盐，想整体换一套随机分布时修改")
    args = ap.parse_args()

    site_root = Path(args.site_root)
    htmls = list(site_root.rglob("*.html"))
    random.shuffle(htmls)

    changed = 0
    for p in htmls:
        try:
            if is_detail_page(p.name):
                if inject_modules(site_root, p, args.modules_per_page, args.salt):
                    changed += 1
        except Exception as e:
            print(f"[WARN] {p}: {e}")
    print(f"✅ nb-variants done. pages changed: {changed}")

if __name__ == "__main__":
    main()
