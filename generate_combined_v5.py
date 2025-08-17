
import os
import glob
import random
import datetime
from pathlib import Path

def generate_image_html(image_path, alt_text, category):
    filename = os.path.basename(image_path)
    return f"""<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>{alt_text}</title>
  <meta name='description' content='{alt_text}'>
  <link rel='canonical' href='{category}/{filename.replace(".jpg", ".html").replace(".png", ".html")}'>
  <script src='../ads.js'></script>
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "ImageObject",
    "contentUrl": "{category}/{filename}",
    "description": "{alt_text}",
    "name": "{filename}"
  }}
  </script>
  <style>body {{ background:#111; color:#fff; text-align:center; font-family:sans-serif; }}
  img {{ max-width:90%; margin-top:20px; border-radius:12px; }}</style>
</head>
<body>
  <h1>{alt_text}</h1>
  <img src='{filename}' alt='{alt_text}'><br>
  <p><a href='../index.html'>← Back to Home</a></p>
  <script src='../ads.js'></script>
</body>
</html>"""

def generate_category_blocks(base_dir):
    blocks = ""
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(d) and not d.startswith(".")]
    for category in categories:
        images = sorted(glob.glob(f"{category}/*.jpg") + glob.glob(f"{category}/*.png"), reverse=True)[:3]
        if not images:
            continue
        block = f"<h2>{category.capitalize()}</h2><div class='gallery'>"
        for img in images:
            block += f"<a href='{img.replace('.jpg', '.html').replace('.png', '.html')}'><img src='{img}'></a>"
        block += f"</div><p><a href='{category}.html'>→ View More</a></p>"
        blocks += f"<div class='section'>{block}</div>\n"
    return blocks

def update_index_html():
    index_file = "index.html"
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            html = f.read()
        blocks = generate_category_blocks(".")
        intro_text = "<p style='text-align:center;'>Welcome to our curated gallery of elegance and style.</p>"
        html = html.replace("<!-- {auto_categories_here} -->", f"{intro_text}\n{blocks}")
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(html)
        print("✅ 已自动更新 index.html 首页分类展示")

def generate_all_images():
    for category in os.listdir():
        if not os.path.isdir(category) or category.startswith("."): continue
        images = glob.glob(f"{category}/*.jpg") + glob.glob(f"{category}/*.png")
        for image in images:
            filename = os.path.basename(image)
            name = filename.rsplit(".", 1)[0].replace("_", " ")
            html = generate_image_html(image, f"{category.capitalize()} image - {name}", category)
            html_path = os.path.join(category, filename.replace(".jpg", ".html").replace(".png", ".html"))
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

if __name__ == "__main__":
    generate_all_images()
    update_index_html()
