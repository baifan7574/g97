import os
from pathlib import Path

def get_category_folders():
    return [
        d for d in os.listdir()
        if os.path.isdir(d) and not d.startswith(".") and d not in ["images", "generator", "keywords"]
    ]

def generate_link_list():
    domain = ""
    if os.path.exists("config.json"):
        import json
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            domain = config.get("domain", "").rstrip("/")

    links = []
    for cat in get_category_folders():
        folder = Path(cat)
        for html_file in folder.glob("*.html"):
            rel_path = f"{cat}/{html_file.name}"
            full_url = f"{domain}/{rel_path}" if domain else rel_path
            links.append(full_url)

    with open("link_list.html", "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        f.write("<title>All Image Pages</title></head><body>\n")
        f.write("<h1>All Image Links</h1>\n<ul>\n")
        for link in sorted(links):
            f.write(f"<li><a href='{link}' target='_blank'>{link}</a></li>\n")
        f.write("</ul>\n</body></html>")

    print("✅ 索引文件 link_list.html 已生成，包含所有图片页面链接。")

if __name__ == "__main__":
    generate_link_list()
