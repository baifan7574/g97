import os
from pathlib import Path
from bs4 import BeautifulSoup

def get_latest_images(category, count=4):
    folder = Path(category)
    if not folder.exists():
        return []
    images = sorted(
        [f for f in folder.glob("*.jpg")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    return images[:count]

def build_category_block(category, images):
    html = f'<div class="category">\n'
    html += f'  <h2>{category.capitalize()}</h2>\n'
    html += f'  <div class="gallery">\n'
    for img in images:
        html += f'    <a data-lightbox="{category}" href="{category}/page1.html">'
        html += f'<img alt="" src="{category}/{img.name}"/></a>\n'
    html += f'  </div>\n'
    html += f'  <div class="view-more"><a href="{category}/page1.html">→ View More</a></div>\n'
    html += f'</div>\n'
    return html

def generate_updated_index():
    index_file = "index.html"
    if not os.path.exists(index_file):
        print("❌ index.html 不存在")
        return

    with open(index_file, "r", encoding="utf-8") as f:
        html = f.read()

    start_marker = "<!-- {auto_categories_here} -->"
    if start_marker not in html:
        print("❌ index.html 缺少 <!-- {auto_categories_here} --> 标记")
        return

    # 扫描所有分类目录
    categories = [
        d for d in os.listdir()
        if os.path.isdir(d) and not d.startswith(".") and d not in ["images", "keywords", "generator"]
    ]

    all_blocks = ""
    for cat in sorted(categories):
        latest_imgs = get_latest_images(cat, count=4)
        if latest_imgs:
            all_blocks += build_category_block(cat, latest_imgs)

    # 替换占位符
    updated_html = html.replace(start_marker, all_blocks)

    with open(index_file, "w", encoding="utf-8") as f:
        f.write(updated_html)

    print("✅ 首页 index.html 已自动插入每类最新4图。")

if __name__ == "__main__":
    generate_updated_index()
