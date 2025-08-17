import json
import os
import math
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime

# ✅ 读取配置文件中的域名
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
domain = config.get("domain", "https://example.com").rstrip("/")
site_name = config.get("site_name", "Image Gallery")

exts = ['.jpg', '.jpeg', '.png', '.webp']

def insert_ads(soup):
    if soup.body:
        ads_code_list = config.get("ads_code", [])
        for ads_html in ads_code_list:
            ads_soup = BeautifulSoup(ads_html, 'html.parser')
            for tag in ads_soup:
                soup.body.append(tag)

def insert_canonical(soup, canonical_url):
    if soup.head:
        existing = soup.find("link", {"rel": "canonical"})
        if existing:
            existing['href'] = canonical_url
        else:
            canonical_tag = soup.new_tag("link", rel="canonical", href=canonical_url)
            soup.head.append(canonical_tag)

def load_keywords(category):
    try:
        with open(f'keywords/{category}.txt', 'r', encoding='utf-8') as f:
            return [kw.strip() for kw in f if kw.strip()]
    except:
        return []

def get_category_folders():
    return [d for d in os.listdir() if os.path.isdir(d) and not d.startswith('.') and d.lower() not in ['images', 'generator', 'keywords']]

def generate_description(keyword):
    return f"{keyword.capitalize()} themed portrait showcasing unique visual storytelling and visual composition."

def generate_paragraph(keyword):
    return (
        f"This portrait highlights the theme of {keyword}, combining aesthetic elements, lighting, and emotional resonance. "
        f"Visitors interested in {keyword} often appreciate unique imagery, artistic expression, and refined visual taste. "
        f"This gallery explores {keyword} concepts through styled visuals and thoughtful design."
    )

def generate_pages_and_images():
    categories = get_category_folders()
    for cat in categories:
        folder = Path(cat)
        images = sorted([f for f in folder.glob('*.jpg')])
        keywords = load_keywords(cat)
        per_page = 20
        total_pages = math.ceil(len(images) / per_page)
        for page in range(total_pages):
            start = page * per_page
            end = start + per_page
            imgs = images[start:end]
            page_file = folder / f'page{page+1}.html'
            with open(page_file, 'w', encoding='utf-8') as f:
                desc = f"Browse {cat} images. Page {page+1} of curated {cat}-style portrait collection."
                f.write(f'<html><head><title>{cat.capitalize()} - Page {page+1}</title><meta name="description" content="{desc}"></head><body>')
                f.write(f'<h1>{cat.capitalize()} Gallery - Page {page+1}</h1><p>{desc}</p>')
                for idx, img_path in enumerate(imgs):
                    name = img_path.stem
                    html_file = folder / f'{name}.html'
                    kw_index = start + idx
                    kw = keywords[kw_index] if kw_index < len(keywords) else cat
                    prev_name = imgs[idx - 1].stem if idx > 0 else ""
                    next_name = imgs[idx + 1].stem if idx < len(imgs) - 1 else ""

                    with open(html_file, 'w', encoding='utf-8') as imgf:
                        imgf.write(f'<html><head><title>{kw}</title>')
                        imgf.write(f'<meta name="description" content="{generate_description(kw)}">')
                        imgf.write(f'<meta name="keywords" content="{kw}">')
                        schema_json = f"""
<script type="application/ld+json">
{{
  "@context": "https://schema.org",
  "@type": "ImageObject",
  "name": "{kw}",
  "contentUrl": "{domain}/{cat}/{img_path.name}",
  "description": "{generate_description(kw)}",
  "author": {{
    "@type": "Organization",
    "name": "{site_name}"
  }}
}}
</script>
"""
                        imgf.write(schema_json)
                        imgf.write('</head><body>')
                        imgf.write(f'<h1>{kw}</h1><img src="{img_path.name}" alt="{kw}" style="max-width:100%"/><br>')
                        imgf.write(f'<p>{generate_paragraph(kw)}</p><div>')
                        if prev_name:
                            imgf.write(f'<a href="{prev_name}.html">Previous</a> | ')
                        if next_name:
                            imgf.write(f'<a href="{next_name}.html">Next</a> | ')
                        imgf.write(f'<a href="page{page+1}.html">Back to List</a> | <a href="../index.html">Home</a></div></body></html>')

                    soup = BeautifulSoup(open(html_file, encoding='utf-8'), 'html.parser')
                    insert_ads(soup)
                    insert_canonical(soup, f"{domain}/{cat}/{name}.html")
                    with open(html_file, 'w', encoding='utf-8') as imgf:
                        imgf.write(str(soup))
                    f.write(f'<a href="{name}.html"><img src="{img_path.name}" width="200"></a>\n')
                f.write('<div style="margin-top:20px">')
                if page > 0:
                    f.write(f'<a href="page{page}.html">Previous</a> ')
                f.write(f'<a href="../index.html">Home</a> ')
                if page < total_pages - 1:
                    f.write(f'<a href="page{page+2}.html">Next</a>')
                f.write('</div></body></html>')

def generate_sitemap():
    with open("sitemap.xml", "w", encoding="utf-8") as sm:
        sm.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        sm.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for cat in get_category_folders():
            folder = Path(cat)
            for file in folder.glob('*.html'):
                file_path = folder / file.name
                lastmod = datetime.utcfromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
                sm.write(f'<url><loc>{domain}/{cat}/{file.name}</loc><lastmod>{lastmod}</lastmod></url>\n')
        sm.write('</urlset>')

def generate_robots_txt():
    content = f"""User-agent: *
Allow: /
Sitemap: {domain}/sitemap.xml
"""
    with open("robots.txt", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == '__main__':
    generate_pages_and_images()
    generate_sitemap()
    generate_robots_txt()
    print("✅ 所有页面与SEO结构生成完毕，包括 canonical、ads、sitemap 和 robots.txt")
