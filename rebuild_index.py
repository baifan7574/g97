import os
import json
from datetime import datetime

# 从 config.json 中读取站点名称和广告代码
def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def generate_homepage():
    config = load_config()
    site_name = config.get("site_name", "My Gallery Site")
    slogan = config.get("slogan", "For Those Who Appreciate More Than Beauty.")
    ads_code = config.get("ads_code", "")

    # HTML 模板
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{site_name}</title>
    <meta name="description" content="{slogan}">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="styles.css" rel="stylesheet">
    <link rel="canonical" href="https://example.com/">
</head>
<body>
    <header>
        <h1>{site_name}</h1>
        <p>{slogan}</p>
    </header>

    <!-- 自动插入图片展示区 -->
    <section class="gallery-container">
        <!-- {{auto_categories_here}} -->
    </section>

    <!-- 广告代码区域 -->
    <div class="ad-zone">
        {ads_code}
    </div>

    <footer>
        <nav>
            <a href="about.html">About</a> |
            <a href="privacy.html">Privacy</a> |
            <a href="contact.html">Contact</a>
        </nav>
        <p>© {datetime.now().year} {site_name}. All rights reserved.</p>
    </footer>
</body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ 首页 index.html 已生成完毕")

if __name__ == "__main__":
    generate_homepage()
