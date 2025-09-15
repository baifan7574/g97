# inject_keywords.py — 多模板 + 可选H1补丁版
from pathlib import Path
import re, html, argparse, shutil, time, random

ROOT = Path(".")
KW_DIR_PRI = ROOT / "selected_keywords"
KW_DIR_FALLBACK = ROOT / "keywords"
MARK_FLAG = "<!-- data-nb-key=\"1\" -->"
BACKUP_SUFFIX = ".bak"

CATEGORIES = [
    "bedroom","dark","fitness","luxury","mirror",
    "office","pages","redroom","shower","soft","uniform"
]

# ===== 模板池（会轮换，降低同质化）=====
TITLE_TEMPLATES = [
    "{kw} | {cat} gallery",
    "{kw} — curated {cat} photos",
    "{cat} · {kw} photography set",
    "{kw} ({cat}) aesthetic portraits",
    "{kw} • {cat} collection",
    "Explore {kw} in {cat} style"
]
DESC_TEMPLATES = [
    "{kw} - {cat} gallery. Aesthetic photography & portraits.",
    "Curated {cat} photos featuring: {kw}. Clean, fast, mobile-ready.",
    "Discover {kw} in our {cat} collection. High quality images.",
    "{cat} theme · {kw}. Handpicked visuals for inspiration.",
    "Selected {cat} portraits — {kw}. Minimal layout, quick view."
]

# 是否在正文里补一个 <h1>{kw}</h1>（有则覆盖第一个，没有则插入）
INJECT_H1 = True

def log(msg): print(time.strftime("[%H:%M:%S] "), msg, flush=True)
def read_text(p: Path) -> str: return p.read_text("utf-8", errors="ignore")
def write_text(p: Path, s: str): p.write_text(s, encoding="utf-8")
def backup_file(fp: Path):
    bak = fp.with_suffix(fp.suffix + BACKUP_SUFFIX)
    if not bak.exists(): shutil.copy2(fp, bak)
def list_html_files(cat_dir: Path):
    return sorted([p for p in cat_dir.glob("*.html") if p.is_file()])

def load_keywords_for(cat: str):
    pri = KW_DIR_PRI / f"{cat}.txt"
    fb  = KW_DIR_FALLBACK / f"{cat}.txt"
    for fp in [pri, fb]:
        if fp.exists():
            arr = [re.sub(r"\s+"," ",x.strip()) for x in read_text(fp).splitlines() if x.strip()]
            seen=set(); out=[]
            for s in arr:
                k=s.lower()
                if k in seen: continue
                seen.add(k); out.append(s)
            return out, str(fp)
    return [], None

# ===== HTML helpers =====
def set_title(html_text: str, new_title: str) -> str:
    esc = html.escape(new_title, quote=False)
    if re.search(r"<title>.*?</title>", html_text, flags=re.I|re.S):
        return re.sub(r"<title>.*?</title>", f"<title>{esc}</title>", html_text, count=1, flags=re.I|re.S)
    return re.sub(r"</head>", f"<title>{esc}</title>\n</head>", html_text, count=1, flags=re.I)

def set_meta_desc(html_text: str, new_desc: str) -> str:
    esc = html.escape(new_desc, quote=True)
    if re.search(r'<meta\s+name=["\']description["\']\s+content=["\'].*?["\']\s*/?>', html_text, flags=re.I|re.S):
        return re.sub(r'<meta\s+name=["\']description["\']\s+content=["\'].*?["\']\s*/?>',
                      f'<meta name="description" content="{esc}" />',
                      html_text, count=1, flags=re.I|re.S)
    return re.sub(r"</head>", f'<meta name="description" content="{esc}" />\n</head>',
                  html_text, count=1, flags=re.I)

def set_first_img_alt(html_text: str, new_alt: str) -> str:
    esc = html.escape(new_alt, quote=True)
    pat = re.compile(r'(<img\b[^>]*?)(?:\s+alt="[^"]*")?([^>]*>)', re.I)
    def repl(m: re.Match) -> str:
        before = re.sub(r'\s+alt="[^"]*"', '', m.group(1), flags=re.I)
        if not before.endswith(' '): before += ' '
        return f'{before}alt="{esc}"{m.group(2)}'
    return pat.sub(repl, html_text, count=1)

def set_h1(html_text: str, kw: str) -> str:
    esc = html.escape(kw, quote=False)
    # 覆盖第一个 <h1>…</h1>
    if re.search(r"<h1[^>]*>.*?</h1>", html_text, flags=re.I|re.S):
        return re.sub(r"<h1[^>]*>.*?</h1>", f"<h1>{esc}</h1>", html_text, count=1, flags=re.I|re.S)
    # 没有 <h1> 的话，优先插在 <main> 里；没有 <main> 就在 <body> 开头
    if re.search(r"<main[^>]*>", html_text, flags=re.I):
        return re.sub(r"(<main[^>]*>)", r"\1\n<h1>"+esc+"</h1>", html_text, count=1, flags=re.I)
    return re.sub(r"(<body[^>]*>)", r"\1\n<h1>"+esc+"</h1>", html_text, count=1, flags=re.I)

def page_has_mark(html_text: str) -> bool:
    return MARK_FLAG in html_text

# ===== 模板选择 =====
def build_title_desc(cat: str, kw: str, seed_idx: int=0):
    # 用页面序号做稳定轮换；不想固定就 random.choice
    t = TITLE_TEMPLATES[seed_idx % len(TITLE_TEMPLATES)]
    d = DESC_TEMPLATES[seed_idx % len(DESC_TEMPLATES)]
    title = t.format(kw=kw, cat=cat)
    desc  = d.format(kw=kw, cat=cat)
    return title, desc

def inject_for_page(html_path: Path, cat: str, kw: str, force: bool=False, idx: int=0) -> str:
    html_text = read_text(html_path)
    if (not force) and page_has_mark(html_text):
        return "skip(marked)"
    title, desc = build_title_desc(cat, kw, seed_idx=idx)
    html_new = set_title(html_text, title)
    html_new = set_meta_desc(html_new, desc)
    html_new = set_first_img_alt(html_new, kw)
    if INJECT_H1:
        html_new = set_h1(html_new, kw)

    if "<head" in html_new.lower():
        html_new = re.sub(r"</head>", f"{MARK_FLAG}\n</head>", html_new, count=1, flags=re.I)
    else:
        html_new = MARK_FLAG + "\n" + html_new

    backup_file(html_path)
    write_text(html_path, html_new)
    return "ok"

def run(force: bool=False):
    cat_dirs = []
    for c in CATEGORIES:
        p = ROOT / c
        if p.exists() and p.is_dir():
            cat_dirs.append((c, p))
    log(f"[cfg] category_dirs -> {[c for c,_ in cat_dirs]}")

    for cat, cat_dir in cat_dirs:
        kws, src = load_keywords_for(cat)
        if not kws:
            log(f"[warn] {cat} :: 无关键词文件（{KW_DIR_PRI}/{KW_DIR_FALLBACK}），跳过")
            continue
        log(f"[kw]   {cat} :: 来自 {src} :: {len(kws)} 条")

        files = list_html_files(cat_dir)
        if not files:
            log(f"[warn] {cat} :: 目录下没有 .html 文件，跳过")
            continue

        # 逐页注入：每页关键词唯一 + 模板轮换
        for i, html_fp in enumerate(files):
            kw = kws[i % len(kws)]
            status = inject_for_page(html_fp, cat, kw, force=force, idx=i)
            log(f"[{status}] {cat} :: {kw} -> {html_fp.relative_to(ROOT)}")

    log("✅ all done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略标记，强制覆盖注入")
    args = ap.parse_args()
    run(force=args.force)
