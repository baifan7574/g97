# v4_content_patch.py —— 纯补丁（不含 v4 功能）
# 只做两件事：
# 1) 仅在页面不合格时修复：<title> 45–60、<meta description> 130–155、正文≥200（多模板+多槽位+稳定随机+目录关键词池）
# 2) 无论是否修内容，都把 canonical 和 JSON-LD 的 url 修正为：domain/相对路径（从 config.json 读取）
# 不做：生成/修改 sitemap、不做 ping、不做上传

import argparse, re, json, hashlib, random, sys
from pathlib import Path
from bs4 import BeautifulSoup

# ===== 可调阈值 =====
TARGET_TITLE = (45, 60)
TARGET_DESC  = (130, 155)
MIN_BODY     = 200

# ===== 槽位词池 =====
STYLES = ["modern","vintage","minimal","urban","cinematic","natural","studio","retro"]
MOODS  = ["elegant","playful","moody","romantic","calm","bold","warm","cool"]
LIGHTS = ["soft lighting","golden-hour glow","window light","neon lights","backlight","overcast"]
COMPS  = ["close-up","rule-of-thirds","symmetry","leading lines","wide shot"]
WEARS  = ["casual","streetwear","office","evening dress","sporty","retro"]
BACKS  = ["urban backdrop","nature scene","indoor studio","minimal set","bedroom scene"]

# ===== 小工具 =====
def _rng(seed: str):
    import hashlib, random
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return random.Random(int(h[:8], 16))

def _facets(r):
    return dict(style=r.choice(STYLES), mood=r.choice(MOODS), light=r.choice(LIGHTS),
                comp=r.choice(COMPS), wear=r.choice(WEARS), back=r.choice(BACKS))

def _clamp(s: str, mx: int):
    if len(s) <= mx: return s
    cut = s[:mx].rsplit(" ", 1)[0]
    return cut if len(cut) >= int(mx*0.8) else s[:mx]

def _pad_to(s: str, mn: int, r, pads: list):
    t = s
    while len(t) < mn and pads:
        t += " — " + r.choice(pads); pads.pop(0)
    return t

def _read_lines(p: Path):
    if not p.exists(): return []
    txt = p.read_text(encoding="utf-8", errors="ignore")
    return [x.strip() for x in txt.splitlines() if x.strip()]

def _site_root_auto():
    cands = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    seen = set()
    for base in cands:
        cur = base
        for _ in range(4):
            if cur in seen: break
            seen.add(cur)
            if (cur/"keywords").exists() or (cur/"index.html").exists() or (cur/"sitemap.xml").exists():
                return cur
            cur = cur.parent
    return Path(__file__).resolve().parent

# ===== DOM 安全 =====
def _ensure_dom(soup: BeautifulSoup):
    if not soup.html: soup.append(soup.new_tag("html"))
    if not soup.head: soup.html.insert(0, soup.new_tag("head"))
    if not soup.body: soup.html.append(soup.new_tag("body"))
    return soup.head, soup.body

def _safe_insert_paragraph(soup: BeautifulSoup, text: str):
    p = soup.new_tag("p"); p.string = text
    h1 = soup.find("h1")
    if h1: h1.insert_after(p); return
    img = soup.find("img")
    if img: img.insert_after(p); return
    _, body = _ensure_dom(soup); body.append(p)

def _body_len(soup: BeautifulSoup):
    return len(re.sub(r"\s+"," ", soup.get_text(" ", strip=True)))

# ===== 关键词池：目录名.txt（无则 all.txt）+ 唯一分配 =====
def _pick_pool_for(rel_dir: str, kw_dir: Path):
    parts = [p.lower() for p in Path(rel_dir).parts if p not in (".","")]
    for name in reversed(parts):
        f = kw_dir / f"{name}.txt"
        if f.exists(): return _read_lines(f)
    return _read_lines(kw_dir / "all.txt")

def _load_used(kw_dir: Path):
    up = kw_dir / "used_keywords.json"
    if up.exists():
        try: return json.loads(up.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

def _save_used(kw_dir: Path, used: dict):
    (kw_dir / "used_keywords.json").write_text(
        json.dumps(used, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def assign_primary_kw(root_dir: Path, abs_filepath: Path):
    kw_dir = root_dir / "keywords"
    kw_dir.mkdir(exist_ok=True)
    used = _load_used(kw_dir)
    site_key = str(root_dir.resolve())
    used.setdefault(site_key, {"map": {}, "used_set": []})

    rel_path = str(abs_filepath.resolve().relative_to(root_dir.resolve()))
    rel_dir  = str(Path(rel_path).parent)

    if rel_path in used[site_key]["map"]:
        return used[site_key]["map"][rel_path]

    pool = _pick_pool_for(rel_dir, kw_dir)
    if not pool:
        miss = root_dir / "logs" / "kw_miss.txt"
        miss.parent.mkdir(parents=True, exist_ok=True)
        miss.open("a", encoding="utf-8").write(rel_path + "\n")
        return None

    r = _rng("kw::"+rel_path)
    start = r.randrange(0, len(pool))
    used_set = set(used[site_key]["used_set"])

    pick = None
    for i in range(len(pool)):
        kw = pool[(start + i) % len(pool)]
        if kw not in used_set: pick = kw; break
    if pick is None: pick = pool[start % len(pool)]  # 词不够允许复用

    used[site_key]["map"][rel_path] = pick
    used[site_key]["used_set"].append(pick)
    _save_used(kw_dir, used)
    return pick

def _infer_kw(soup: BeautifulSoup, filepath: Path):
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True): return h1.get_text(strip=True)
    return filepath.stem.replace("_"," ").replace("-"," ").strip()

# ===== 从 config.json 读取 domain；修正 canonical / JSON-LD =====
def _read_domain(root_dir: Path):
    cfg = root_dir / "config.json"
    if not cfg.exists(): return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        d = (data.get("domain") or "").strip().rstrip("/")
        return d or None
    except Exception:
        return None

def fix_canonical_and_schema(soup: BeautifulSoup, filepath: Path, root_dir: Path):
    """把 canonical 与 JSON-LD 的 url 修成 domain/相对路径（保留子目录）。读不到 domain 则跳过。"""
    domain = _read_domain(root_dir)
    if not domain: return False
    rel = str(filepath.resolve().relative_to(root_dir.resolve())).replace("\\", "/")
    expected = f"{domain}/{rel}"
    changed = False

    head, _ = _ensure_dom(soup)
    canon = soup.find("link", {"rel": "canonical"})
    if not canon:
        canon = soup.new_tag("link", rel="canonical", href=expected)
        head.append(canon); changed = True
    elif canon.get("href") != expected:
        canon["href"] = expected; changed = True

    updated = False
    for sc in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            obj = json.loads(sc.string or "")
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("@type") in ("WebPage","CollectionPage","ItemPage"):
            if obj.get("url") != expected:
                obj["url"] = expected
                sc.string = json.dumps(obj, ensure_ascii=False)
                changed = True
            updated = True
            break
    if not updated:
        schema = {"@context":"https://schema.org","@type":"WebPage",
                  "name": filepath.stem, "url": expected}
        sc2 = soup.new_tag("script", type="application/ld+json")
        sc2.string = json.dumps(schema, ensure_ascii=False)
        head.append(sc2); changed = True
    return changed

# ===== 文案生成 =====
def gen_title(keyword: str, brand: str, seed: str):
    r = _rng("t::"+seed); f = _facets(r)
    cands = [
        f"{keyword} {f['style']} portraits | {brand}",
        f"{keyword} gallery — {f['mood']} tone | {brand}",
        f"{keyword} photos, {f['light']} | {brand}",
        f"High-quality {keyword} images — {f['comp']} | {brand}",
        f"{keyword} {f['mood']} lookbook | {brand}",
    ]
    t = r.choice(cands)
    if len(t) < TARGET_TITLE[0]: t += f" — {f['mood']} {f['style']}"
    return _clamp(t, TARGET_TITLE[1])

def gen_desc(keyword: str, seed: str):
    r = _rng("d::"+seed); f = _facets(r)
    base = (f"Explore {keyword} in {f['style']} style with {f['mood']} vibe, "
            f"{f['light']}, and {f['comp']} framing. Curated images on a fast, clean page.")
    pads = [f"{f['wear']} looks and {f['back']}",
            "Simple navigation helps discovery",
            "Mobile-friendly layout for smooth viewing",
            "Short notes keep context clear",
            "Clean typography keeps focus on the visuals"]
    s = _pad_to(base, TARGET_DESC[0], r, pads)
    return _clamp(s, TARGET_DESC[1])

def gen_para(keyword: str, seed: str):
    r = _rng("p::"+seed); f = _facets(r)
    s = (f"This set explores {keyword} through {f['style']} aesthetics and {f['mood']} tone under {f['light']}. "
         f"Compositions use {f['comp']} with {f['back']}, keeping focus clear and tidy. "
         f"Details like {f['wear']} styling and balanced colors make browsing easy.")
    if len(s) < MIN_BODY:
        s += " Pages load quickly and related links help deeper viewing."
    return _clamp(s, 300)

# ===== 只修不合格的内容 =====
def enhance_content_if_needed(soup: BeautifulSoup, filepath: Path, brand: str, root_dir: Path):
    title_now = (soup.title.string if soup.title and soup.title.string else "")
    mdesc = soup.find("meta", {"name":"description"})
    desc_now = (mdesc.get("content") if mdesc else "") or ""
    body_len = _body_len(soup)

    need = (len(title_now) < 30) or (len(desc_now) < 110) or (body_len < MIN_BODY)
    if not need: return False

    kw = assign_primary_kw(root_dir, filepath) or _infer_kw(soup, filepath)
    seed = str(filepath)

    new_title = gen_title(kw, brand, seed)
    new_desc  = gen_desc(kw, seed)
    new_para  = gen_para(kw, seed)

    head, _ = _ensure_dom(soup)

    if not soup.title:
        t = soup.new_tag("title"); head.append(t)
    soup.title.string = new_title

    mdesc = soup.find("meta", {"name":"description"})
    if not mdesc:
        mdesc = soup.new_tag("meta", attrs={"name":"description","content":new_desc})
        head.append(mdesc)
    else:
        mdesc["content"] = new_desc

    if body_len < MIN_BODY:
        _safe_insert_paragraph(soup, new_para)

    return True

# ===== 主函数 =====
def main():
    ap = argparse.ArgumentParser(description="纯补丁：内容只修不合格 + 修正 canonical/JSON-LD（零参数可跑）")
    ap.add_argument("--root", help="站点根目录，不填自动识别")
    ap.add_argument("--brand", help="品牌/站名，不填用根目录名")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else _site_root_auto()
    brand = args.brand if args.brand else root.name
    if not root.exists(): print(f"[FATAL] 根目录不存在：{root}"); sys.exit(2)

    (root/"logs").mkdir(exist_ok=True)
    html_files = list(root.rglob("*.html"))
    fixed_content = fixed_canonical = 0

    for i, fp in enumerate(html_files, 1):
        try:
            html = fp.read_text(encoding="utf-8")
        except Exception:
            html = fp.read_text(errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        if enhance_content_if_needed(soup, fp, brand, root):
            fixed_content += 1

        if fix_canonical_and_schema(soup, fp, root):
            fixed_canonical += 1

        new_html = str(soup)
        if new_html != html:
            fp.write_text(new_html, encoding="utf-8")

        if i % 500 == 0:
            print(f"[PROGRESS] {i}/{len(html_files)} ; content_fixed={fixed_content} ; canonical_fixed={fixed_canonical}")

    print(f"[DONE] root={root} ; total={len(html_files)} ; content_fixed={fixed_content} ; canonical_fixed={fixed_canonical}")

if __name__ == "__main__":
    main()
