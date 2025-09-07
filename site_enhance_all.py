# -*- coding: utf-8 -*-
"""
site_enhance_all.py  (2025-09-07)
功能集合：
1) 首页 Slogan 随机化
2) 分类页差异化描述（Spintax）
3) 结构差异化：专题页(/tags) + 分类页 Top Collections + 单图页相关推荐
4) 修复点：
   - CSS 仅注入 <head>（无“样式当正文”）
   - 插入点以完整 <img ...> 为边界（不截断标签）
   - 所有内链改为相对路径（本地 file:/// 预览正常；线上亦兼容）
   - 专题命中：优先 <img alt> 和 <title>，无则回退文件名
   - 专题无内容不生成、不显示（避免空链接）
5) 分类目录自动发现（无需手填）

【运行顺序建议】
生成HTML/图片 -> SEO注入 -> 运行本脚本 -> 广告注入 -> 部署 -> 提交Sitemap/IndexNow
"""

import os, re, json, random, math
from pathlib import Path

ROOT = Path(".")
HTML_EXTS = (".html", ".htm")

CFG_PATH = ROOT / "site_structure_config.json"
SLOGANS_PATH = ROOT / "slogans.txt"
CAT_TMPL_PATH = ROOT / "category_desc_templates.txt"

DEFAULT_CFG = {
    "variant": "auto",               # auto | A | B | C
    "category_dirs": [],            # 为空则自动发现
    # 专题集合（可以在 site_structure_config.json 中改 include 关键词）
    "collections": [
        {"slug": "morning-light", "title": "Morning Light", "include": ["morning","sun","window","soft"]},
        {"slug": "cozy-bedroom",  "title": "Cozy Bedroom",  "include": ["blanket","cozy","warm","pillow","bed"]},
        {"slug": "soft-portraits","title": "Soft Portraits","include": ["soft","bokeh","gentle","pastel"]},
        {"slug": "moody-tones",   "title": "Moody Tones",   "include": ["dark","shadow","night","lamp"]},
        {"slug": "natural-look",  "title": "Natural Look",  "include": ["natural","casual","smile","clean"]}
    ],
    # 相关推荐固定展示张数（你要 3 张）
    "related_thumbs": 3,
    "dry_run": False
}

CSS_BLOCK = """
<style>
.nb-wrap{font:14px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial;color:#ddd}
.nb-h2{font-size:20px;margin:16px 0 8px}
.nb-tags{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.nb-tag a{display:inline-block;padding:6px 10px;border-radius:999px;border:1px solid #444;text-decoration:none}
.nb-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.nb-thumb{display:block;border-radius:8px;overflow:hidden}
.nb-thumb img{width:100%;height:auto;display:block}
.nb-box{background:#111;border:1px solid #2a2a2a;border-radius:12px;padding:12px;margin:14px 0}
.nb-note{opacity:.8;margin-top:8px}
.nb-btm{margin:20px 0}
.nb-breadcrumb{font:13px/1.6 system-ui;margin:8px 0 10px}
.nb-breadcrumb a{text-decoration:none}
.nb-breadcrumb a:hover{text-decoration:underline}
@media (min-width:1024px){
  .nb-sidebar{float:right;width:280px;margin-left:16px}
}
</style>
"""

MARK = {
    "variant": "data-nb-variant",
    "collections": "data-nb-collections",
    "related": "data-nb-related",
    "slogan": "data-nb-slogan",
    "catdesc": "data-nb-catdesc"
}

# ------------------- 基础加载与工具 -------------------
def load_cfg():
    if CFG_PATH.exists():
        try:
            user = json.loads(CFG_PATH.read_text("utf-8"))
            return {**DEFAULT_CFG, **user}
        except Exception:
            return DEFAULT_CFG
    return DEFAULT_CFG

def autodiscover_categories(root: Path):
    """根目录下一层凡是包含 page*.html 或 index.html 的目录，视为分类目录。"""
    dirs = set()
    for p in root.iterdir():
        if not p.is_dir(): 
            continue
        has_cat = any(q.name.lower().startswith("page") and q.suffix.lower() in (".html",".htm")
                      for q in p.glob("*.htm*"))
        has_idx = (p / "index.html").exists() or (p / "index.htm").exists()
        if has_cat or has_idx:
            dirs.add(p.name)
    return sorted(dirs)

CFG = load_cfg()
if not CFG.get("category_dirs"):
    CFG["category_dirs"] = autodiscover_categories(ROOT)
DRY = CFG.get("dry_run", False)
print("[cfg] category_dirs ->", CFG["category_dirs"])

def safe_write(p: Path, new_text: str):
    if DRY:
        print("[DRY] would write:", p); return
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        try: bak.write_text(p.read_text("utf-8", errors="ignore"), "utf-8")
        except Exception: pass
    p.write_text(new_text, "utf-8")

def insert_css_once(html: str):
    """确保 CSS 注入到 <head>（找不到则退到 <body> 后）。"""
    if re.search(r"\.nb-wrap\{", html):  # 已注入
        return html
    m = re.search(r"</head>", html, flags=re.I)
    if m:
        return html[:m.start()] + CSS_BLOCK + "\n" + html[m.start():]
    m = re.search(r"<body[^>]*>", html, flags=re.I)
    if m:
        return html[:m.end()] + "\n" + CSS_BLOCK + html[m.end():]
    return CSS_BLOCK + html

def insert_block(html: str, block: str, pos="below"):
    """以完整 <img ...> 标签边界作为插入点，避免截断标签。"""
    if not block: return html, False
    html = insert_css_once(html)

    imgs = list(re.finditer(r"<img\b[^>]*>", html, flags=re.I))
    def after_main_or_body(s):
        m = re.search(r"<main[^>]*>", s, flags=re.I)
        if m: return m.end()
        m = re.search(r"<body[^>]*>", s, flags=re.I)
        return m.end() if m else None

    if pos == "above":
        idx = imgs[0].start() if imgs else after_main_or_body(html)
    elif pos == "below":
        idx = imgs[-1].end() if imgs else after_main_or_body(html)
    elif pos == "sidebar":
        m = re.search(r"<body[^>]*>", html, flags=re.I)
        idx = m.end() if m else 0
        block = block.replace('class="nb-box"', 'class="nb-box nb-sidebar"', 1)
    else:
        idx = after_main_or_body(html)

    if idx is None: return html, False
    out = html[:idx] + "\n" + block + "\n" + html[idx:]
    return out, True

def rel_href(from_file: Path, to_file: Path):
    """生成相对链接（/ 与 \ 统一为 /），支持本地 file:/// 预览与线上部署。"""
    rel = os.path.relpath(to_file.as_posix(), start=from_file.parent.as_posix())
    return rel.replace("\\", "/")

# ------------------- 首页 Slogan -------------------
def load_slogans():
    if SLOGANS_PATH.exists():
        arr = [x.strip() for x in SLOGANS_PATH.read_text("utf-8", errors="ignore").splitlines() if x.strip()]
        if arr: return arr
    return [
        "Images that whisper stories.",
        "A quiet gallery of light and mood.",
        "Elegance in subtle frames.",
        "Where portraits meet emotion.",
        "Gentle tones, honest moments."
    ]

def patch_home_slogan():
    idx = ROOT / "index.html"
    if not idx.exists(): print("[slogan] skip (no index.html)"); return
    html = idx.read_text("utf-8", errors="ignore")
    if MARK["slogan"] in html: print("[slogan] exists ->", idx); return
    s = random.choice(load_slogans())
    m = re.search(r"<h1[^>]*>.*?</h1>", html, flags=re.I|re.S)
    if not m:
        m = re.search(r"<body[^>]*>", html, flags=re.I)
        if not m: print("[slogan] no insert point"); return
    insert_at = m.end()
    out = html[:insert_at] + f'\n<p class="nb-sub" {MARK["slogan"]}="1" style="opacity:.9;margin:6px 0 12px">{s}</p>\n' + html[insert_at:]
    safe_write(idx, out); print("[slogan] ok ->", idx)

# ------------------- 分类页描述 -------------------
def load_cat_templates():
    base = [
        "Explore {bedroom|gallery|collection} portraits with {soft|gentle|subtle} lighting and {natural|casual} moods. This page curates {intimate|quiet|serene} moments and {elegant|clean} frames.",
        "Browse {bedroom|portrait} images featuring {morning light|cozy atmosphere|warm tones} and {relaxed|honest} expressions. A {handpicked|curated} set for {inspiration|reference}.",
        "Discover {bedroom-themed|indoor} portraits focused on {texture|light|color} and {emotion|presence}. Crafted for viewers who enjoy {subtle|minimal} aesthetics."
    ]
    if CAT_TMPL_PATH.exists():
        arr = [x.strip() for x in CAT_TMPL_PATH.read_text("utf-8", errors="ignore").splitlines() if x.strip()]
        if arr: base = arr
    return base

def spintax(s: str):
    while True:
        m = re.search(r"\{([^{}]+)\}", s)
        if not m: break
        s = s[:m.start()] + random.choice(m.group(1).split("|")) + s[m.end():]
    return s

def is_category_page(p: Path):
    parts = p.relative_to(ROOT).parts
    if len(parts) < 2: return False
    cat = parts[-2].lower()
    if cat not in [c.lower() for c in CFG["category_dirs"]]: return False
    n = p.name.lower()
    return n.startswith("page") or n in ("index.html","index.htm")

def insert_cat_desc(p: Path):
    html = p.read_text("utf-8", errors="ignore")
    if MARK["catdesc"] in html: return False, "exists"
    newp = f'<p {MARK["catdesc"]}="1">{spintax(random.choice(load_cat_templates()))}</p>'
    h1 = re.search(r"<h1[^>]*>.*?</h1>", html, flags=re.I|re.S)
    if h1:
        after = html[h1.end():]
        mp = re.search(r"<p[^>]*>.*?</p>", after, flags=re.I|re.S)
        if mp:
            seg = after[mp.start():mp.end()]
            if len(seg) < 260 and re.search(r"\b(browse|gallery|collection|images|curated)\b", seg, flags=re.I):
                out = html[:h1.end()] + after[:mp.start()] + newp + after[mp.end():]
                safe_write(p, out); return True, "replaced"
        out = html[:h1.end()] + "\n" + newp + "\n" + html[h1.end():]
        safe_write(p, out); return True, "inserted"
    m = re.search(r"<main[^>]*>", html, flags=re.I) or re.search(r"<body[^>]*>", html, flags=re.I)
    if not m: return False, "no-insert-point"
    out = html[:m.end()] + "\n" + newp + "\n" + html[m.end():]
    safe_write(p, out); return True, "inserted"

def patch_category_descriptions():
    touched = 0; total = 0
    for root, _, files in os.walk(ROOT):
        for f in files:
            if not f.lower().endswith(HTML_EXTS): continue
            p = Path(root) / f
            if is_category_page(p):
                total += 1
                ok, msg = insert_cat_desc(p)
                if ok: touched += 1
                print(f"[catdesc] {msg} -> {p}")
    print(f"[catdesc] done: {touched}/{total}")

# ------------------- 专题/相关推荐/A/B/C -------------------
def pick_variant():
    v = CFG.get("variant","auto")
    return v if v in ("A","B","C") else random.choice(["A","B","C"])

def is_image_page(p: Path):
    parts = p.relative_to(ROOT).parts
    if len(parts) < 2: return False
    cat = parts[-2].lower()
    if cat not in [c.lower() for c in CFG["category_dirs"]]: return False
    n = p.name.lower()
    return n.endswith((".html",".htm")) and not n.startswith("page") and n not in ("index.html","index.htm")

def first_img_src(html: str):
    m = re.search(r'<img[^>]+src="([^"]+)"', html, flags=re.I)
    return m.group(1) if m else ""

def list_image_pages(cat: str):
    d = ROOT / cat
    if not d.exists(): return []
    out = []
    for p in sorted(d.rglob("*.htm*")):
        if is_image_page(p): out.append(p)
    return out

def page_keywords_for_match(html_text: str, file_name: str):
    """用于专题命中：alt/title 优先，文件名回退。"""
    alts = re.findall(r'<img[^>]+alt="([^"]+)"', html_text, flags=re.I)
    title = ""
    m = re.search(r"<title>(.*?)</title>", html_text, flags=re.I|re.S)
    if m: title = re.sub(r"\s+"," ", m.group(1)).strip()
    text = " ".join(alts + [title, file_name.lower()])
    return text.lower()

# —— 相对链接版本相关推荐（固定 3 张）
def build_related_block(curr_page_path: Path, cat: str, exclude_name: str):
    pages = [p for p in list_image_pages(cat) if p.name != exclude_name]
    random.shuffle(pages)
    k = min(max(CFG.get("related_thumbs", 3), 0), len(pages))  # 固定 3 张；不足则全用
    if k <= 0: return ""
    thumbs = []
    for p in pages[:k]:
        html = p.read_text("utf-8", errors="ignore")
        src = first_img_src(html)
        href = rel_href(curr_page_path, p)
        # 尝试把以 / 开头的 src 也转相对（若能定位文件）
        if src.startswith("/"):
            src_path = (p.parent / src.lstrip("/")).resolve()
            if src_path.exists():
                src = rel_href(curr_page_path, src_path)
        thumbs.append(f'<a class="nb-thumb" href="{href}"><img loading="lazy" src="{src}" alt=""></a>')
    return f'''
<div class="nb-box" {MARK["related"]}="1">
  <div class="nb-h2">You may also like</div>
  <div class="nb-grid">{''.join(thumbs)}</div>
</div>
'''.strip()

# 统计每个专题命中数，用于分类页过滤空专题
TAG_COUNTS = {}  # slug -> count

def ensure_tags_pages():
    global TAG_COUNTS
    TAG_COUNTS = {}
    tags_dir = ROOT / "tags"
    if not DRY: tags_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for c in CFG["collections"]:
        slug, title, inc = c["slug"], c["title"], [x.lower() for x in c["include"]]
        matches = []
        for cat in CFG["category_dirs"]:
            for p in list_image_pages(cat):
                html = p.read_text("utf-8", errors="ignore")
                text = page_keywords_for_match(html, p.name)
                if any(k in text for k in inc):
                    matches.append(p)  # Path 对象
        TAG_COUNTS[slug] = len(matches)
        if len(matches) == 0:
            print(f"[tags] {slug}: 0 items (skip gen)")
            continue  # 空专题不生成页面

        per = 30
        pages = max(1, math.ceil(len(matches)/per))
        for i in range(pages):
            cur = matches[i*per:(i+1)*per]
            # 在 /tags/ 页面里，指向图片页与图片 src 都用相对路径
            base_out = tags_dir / (f"{slug}.html" if i==0 else f"{slug}_{i+1}.html")
            thumbs = []
            for p in cur:
                html = p.read_text("utf-8", errors="ignore")
                item_href = rel_href(base_out, p)
                src = first_img_src(html)
                if src.startswith("/"):
                    src_path = (p.parent / src.lstrip("/")).resolve()
                    if src_path.exists():
                        src = rel_href(base_out, src_path)
                thumbs.append(f'<a class="nb-thumb" href="{item_href}"><img loading="lazy" src="{src}" alt=""></a>')
            nav = []
            if i+1 < pages:
                nav.append(f'<a href="{f"{slug}_{i+2}.html"}">Next</a>')
            home_href = rel_href(base_out, ROOT / "index.html")
            content = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title} Collection</title>{CSS_BLOCK}</head>
<body class="nb-wrap">
  <nav class="nb-breadcrumb"><a href="{home_href}">Home</a> › <span aria-current="page">{title}</span></nav>
  <h1>{title} Collection</h1>
  <div class="nb-grid">{''.join(thumbs)}</div>
  <div class="nb-btm">{' | '.join(nav)}</div>
</body></html>"""
            if DRY:
                print("[DRY] tags ->", base_out, "items:", len(cur))
            else:
                base_out.write_text(content, "utf-8")
        total += len(matches)
        print(f"[tags] {slug}: {len(matches)} items, pages={pages}")
    print(f"[tags] total items: {total}")

def build_collections_links(curr_page_path: Path, collections_counts: dict):
    links = []
    for c in CFG["collections"]:
        slug = c["slug"]
        if collections_counts.get(slug, 0) <= 0:
            continue  # 空专题不显示
        tag_file = ROOT / "tags" / f"{slug}.html"
        href = rel_href(curr_page_path, tag_file)
        links.append(f'<span class="nb-tag"><a href="{href}">{c["title"]}</a></span>')
    if not links:
        return ""
    return f'''
<div class="nb-box" {MARK["collections"]}="1">
  <div class="nb-h2">Top Collections</div>
  <div class="nb-tags">{''.join(links)}</div>
</div>
'''.strip()

def patch_category_page(p: Path, variant: str):
    html = p.read_text("utf-8", errors="ignore")
    changed = False
    if MARK["collections"] not in html:
        blk = build_collections_links(p, TAG_COUNTS)
        if blk:
            pos = {"A":"above","B":"sidebar","C":"below"}[variant]
            html, ok = insert_block(html, blk, pos=pos)
            if ok: changed = True
    if changed:
        html = html.replace("<body", f'<body {MARK["variant"]}="{variant}"', 1)
        safe_write(p, html); print("[category]", variant, "->", p)
    else:
        print("[category] skip ->", p)

def patch_image_page(p: Path, variant: str):
    cat = p.relative_to(ROOT).parts[-2]
    html = p.read_text("utf-8", errors="ignore")
    if MARK["related"] in html: print("[image] exists ->", p); return
    blk = build_related_block(p, cat, exclude_name=p.name)
    if not blk: print("[image] no-related ->", p); return
    pos = {"A":"below","B":"sidebar","C":"below"}[variant]
    html, ok = insert_block(html, blk, pos=pos)
    if ok:
        html = html.replace("<body", f'<body {MARK["variant"]}="{variant}"', 1)
        safe_write(p, html); print("[image]", variant, "->", p)
    else:
        print("[image] no-insert-point ->", p)

def run_structure_patch():
    variant = pick_variant()
    print("Variant:", variant)
    ensure_tags_pages()
    for root, _, files in os.walk(ROOT):
        for f in files:
            if not f.lower().endswith(HTML_EXTS): continue
            p = Path(root) / f
            if is_category_page(p):
                patch_category_page(p, variant)
            elif is_image_page(p):
                patch_image_page(p, variant)

def main():
    patch_home_slogan()
    patch_category_descriptions()
    run_structure_patch()
    print("\n✅ enhance done.")

if __name__ == "__main__":
    main()
