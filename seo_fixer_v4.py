from pathlib import Path
from bs4 import BeautifulSoup
import random, datetime, json, os, requests

base_path = Path.cwd()
html_files = list(base_path.rglob("*.html"))
total_fixed = 0

log_file = open("seo_fixer_log.txt", "w", encoding="utf-8")

# ===== 读取 config.json，获取域名 =====
domain = "https://example.com"  # 默认值
config_path = base_path / "config.json"
if config_path.exists():
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        domain = cfg.get("domain", domain).rstrip("/")  # 去掉末尾斜杠
    except Exception as e:
        log_file.write(f"[WARN] 读取 config.json 失败: {e}\n")

# ===== 读取关键词文件，用于生成长文本 =====
keywords_dir = base_path / "keywords"
keywords_pool = []
if keywords_dir.exists():
    for f in keywords_dir.glob("*.txt"):
        words = f.read_text(encoding="utf-8").splitlines()
        keywords_pool.extend([w.strip() for w in words if len(w.strip()) > 2])
if not keywords_pool:
    keywords_pool = ["photo", "gallery", "collection", "visual", "image"]  # 备用关键词

# ===== 生成长文本补丁 =====
def generate_random_text(keyword="photo"):
    chosen = random.sample(keywords_pool, min(5, len(keywords_pool)))
    templates = [
        f"This {keyword} gallery explores {', '.join(chosen)} in depth, offering fresh perspectives.",
        f"Our {keyword} collection integrates {', '.join(chosen)}, frequently updated with new visuals.",
        f"High-quality {keyword} images connected with {', '.join(chosen)} for diverse inspiration.",
        f"Explore this {keyword} showcase, combining {', '.join(chosen)} to enrich user experience."
    ]
    return " ".join([random.choice(templates) for _ in range(2)])

# ===== 分类页长文本补丁 =====
def add_category_text(soup, file):
    name = file.stem
    if name.startswith("page"):   # 判断是 pageN.html
        if len(soup.get_text()) < 150:
            p = soup.new_tag("p")
            p.string = (
                f"This is {name}, part of our curated gallery collection. "
                f"Each page highlights unique themes, aesthetics, and visual styles, "
                f"helping visitors explore different categories with richer context and inspiration."
            )
            soup.body.append(p)
            return True
    return False

# ===== 内链补丁（优先同目录） =====
def add_internal_links(soup, all_files, current_file):
    same_dir = [f for f in all_files if f.parent == current_file.parent and f != current_file]
    candidates = same_dir if same_dir else [f for f in all_files if f != current_file]
    if not candidates:
        return
    related = random.sample(candidates, min(3, len(candidates)))
    div = soup.new_tag("div")
    div.string = "More related: "
    for f in related:
        a = soup.new_tag("a", href=f.name)
        a.string = f.name.replace(".html", "")
        div.append(a)
        div.append(" | ")
    soup.body.append(div)

# ===== 更新 sitemap.xml 的 lastmod =====
def update_sitemap():
    sitemap_path = base_path / "sitemap.xml"
    if not sitemap_path.exists():
        return
    text = sitemap_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    for line in text:
        if "<lastmod>" in line:
            new_lines.append(f"    <lastmod>{today}</lastmod>")
        else:
            new_lines.append(line)
    sitemap_path.write_text("\n".join(new_lines), encoding="utf-8")
    # 提交到 Google
    try:
        ping_url = f"https://www.google.com/ping?sitemap={domain}/sitemap.xml"
        requests.get(ping_url, timeout=10)
        log_file.write(f"[PING] 提交 sitemap 到 Google: {ping_url}\n")
    except Exception as e:
        log_file.write(f"[WARN] 提交 sitemap 失败: {e}\n")

# ===== 删除无效页面 =====
def remove_invalid(file):
    try:
        html = file.read_text(encoding="utf-8")
        if len(html.strip()) == 0 or "window.location.href" in html:
            file.unlink()
            log_file.write(f"[DEL] {file} (empty or redirect)\n")
            return True
    except:
        return False
    return False

# ===== 主循环，逐个修复 HTML =====
for file in html_files:
    if remove_invalid(file):
        continue

    try:
        html = file.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        # ==== 去重：删除旧 canonical / schema ====
        for old_tag in soup.find_all("link", {"rel": "canonical"}):
            old_tag.decompose()
        for old_tag in soup.find_all("script", {"type": "application/ld+json"}):
            old_tag.decompose()

        # <title>
        if not soup.title:
            title = soup.new_tag("title")
            title.string = file.stem
            soup.head.append(title)

        # meta description
        if not soup.find("meta", {"name": "description"}):
            desc = soup.new_tag("meta", attrs={"name": "description", "content": f"{file.stem} photo collection and gallery"})
            soup.head.append(desc)

        # canonical
        canonical = soup.new_tag("link", rel="canonical", href=f"{domain}/{file.name}")
        soup.head.append(canonical)

        # schema (JSON-LD)
        schema = {
            "@context": "https://schema.org",
            "@type": "WebPage",
            "name": file.stem,
            "url": f"{domain}/{file.name}"
        }
        script = soup.new_tag("script", type="application/ld+json")
        script.string = json.dumps(schema)
        soup.head.append(script)

        # img alt
        for img in soup.find_all("img"):
            if not img.get("alt"):
                img["alt"] = file.stem

        # 普通页面长文本补丁
        if len(soup.get_text()) < 200:
            p = soup.new_tag("p")
            p.string = generate_random_text(file.stem)
            soup.body.append(p)
            log_file.write(f"[TEXT] Added paragraph to {file}\n")

        # 分类页长文本补丁
        if add_category_text(soup, file):
            log_file.write(f"[CAT] Added category text to {file}\n")

        # 内链补丁
        add_internal_links(soup, html_files, file)

        # 写回文件
        file.write_text(str(soup), encoding="utf-8")
        total_fixed += 1
        log_file.write(f"[OK] {file}\n")

    except Exception as e:
        log_file.write(f"[ERROR] {file}: {e}\n")

# ===== 更新 sitemap.xml 并通知 Google =====
update_sitemap()

log_file.close()
print(f"[OK] 共修复页面：{total_fixed} 个 ✅")
