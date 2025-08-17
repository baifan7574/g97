import json
import re

# 读取配置文件
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

site_name = config.get("site_name", "Gentleman’s Frame")
domain = config.get("domain", "https://example.com")
ads_code_raw = config.get("ads_code", "")
ads_code = ads_code_raw[0].strip() if isinstance(ads_code_raw, list) else str(ads_code_raw).strip()


# 加载 index.html
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 替换 <title> 内容
html = re.sub(r"<title>.*?</title>", f"<title>{site_name}</title>", html, flags=re.DOTALL)

# 替换 <h1> 内容
html = re.sub(r"<h1>.*?</h1>", f"<h1>{site_name}</h1>", html)

# 替换 <p> 副标题（如果有的话）
html = re.sub(r"<p>.*?</p>", f"<p>For Men Who Appreciate More Than Beauty.</p>", html)

# 插入 canonical 标签（如果不存在）
if 'rel="canonical"' not in html:
    html = html.replace(
        "</head>",
        f'<link rel="canonical" href="{domain}/">\n</head>'
    )

# 替换 ads.js 加载方式
html = re.sub(r'<script\s+src=["\']ads\.js["\']>\s*</script>', ads_code, html)

# 插入 Schema.org 结构（如果没有）
if 'application/ld+json' not in html:
    schema_json = f"""
<script type="application/ld+json">
{{
  "@context": "https://schema.org",
  "@type": "WebSite",
  "name": "{site_name}",
  "url": "{domain}/",
  "description": "For men who appreciate more than beauty. A curated gallery of elegance."
}}
</script>
"""
    html = html.replace("</head>", schema_json + "\n</head>")

# 保存修改后文件
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ 补丁应用成功：标题 / canonical / ads / schema 全部已插入 index.html")
