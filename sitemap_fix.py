# -*- coding: utf-8 -*-
import os, json, datetime

ROOT = os.path.abspath(os.path.dirname(__file__))

def load_domain():
    cfg = os.path.join(ROOT, "config.json")
    with open(cfg, "r", encoding="utf-8") as f:
        d = json.load(f).get("domain", "").strip()
    if d.endswith("/"): d = d[:-1]
    if not d.startswith("http"):
        raise SystemExit(f"[FATAL] config.json 的 domain 非法: {d}")
    return d

def should_skip(path):
    p = path.replace("\\","/").lower()
    # 跳过这几类目录/文件
    skip_dirs = ("/generator/", "/logs/", "/keywords/", "/.git/")
    if any(s in p for s in skip_dirs): return True
    skip_files = ("template",)  # 跳过 *_template.html
    if any(s in os.path.basename(p) for s in skip_files): return True
    return False

def iter_html(root):
    for dp, dn, fn in os.walk(root):
        dpn = dp.replace("\\","/")
        if any(x in dpn.lower() for x in ["/generator", "/logs", "/keywords", "/.git"]):
            continue
        for name in fn:
            if name.lower().endswith(".html"):
                full = os.path.join(dp, name)
                if not should_skip(full):
                    yield full

def fmt_date(ts):
    # 谷歌接受 yyyy-MM-dd（也可用 ISO8601）
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

def build_sitemap(domain):
    urls = []
    for f in iter_html(ROOT):
        rel = os.path.relpath(f, ROOT).replace("\\","/")
        loc = f"{domain}/{rel}"
        last = fmt_date(os.path.getmtime(f))
        urls.append((loc, last))
    return urls

def write_xml(urls):
    out = os.path.join(ROOT, "sitemap.xml")
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    for loc, last in urls:
        lines.append("  <url>")
        lines.append(f"    <loc>{loc}</loc>")
        lines.append(f"    <lastmod>{last}</lastmod>")
        lines.append("  </url>")
    lines.append("</urlset>")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"[OK] 生成 sitemap.xml ：{len(urls)} 条")

if __name__ == "__main__":
    dom = load_domain()
    urls = build_sitemap(dom)
    if not urls:
        raise SystemExit("[FATAL] 没找到任何 HTML，确认目录/过滤规则")
    write_xml(urls)
