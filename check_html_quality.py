
import os
import re

# 设置要检测的 HTML 文件目录，请修改为你的实际路径
TARGET_DIR = r"D:/项目/girl"

def check_html_file(filepath):
    result = {
        "filename": os.path.basename(filepath),
        "has_title": False,
        "has_description": False,
        "has_alt": False,
        "content_length": 0,
        "has_meta_refresh": False,
        "has_js_redirect": False
    }

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        result["has_title"] = bool(re.search(r'<title>.*?</title>', content, re.IGNORECASE))
        result["has_description"] = bool(re.search(r'<meta\s+name=[\'\"]description[\'\"]\s+content=[\'\"]', content, re.IGNORECASE))
        result["has_alt"] = bool(re.search(r'<img[^>]+alt=[\'\"]', content, re.IGNORECASE))
        result["content_length"] = len(re.sub(r'<[^>]+>', '', content))  # 去掉所有HTML标签
        result["has_meta_refresh"] = bool(re.search(r'<meta\s+http-equiv=[\'\"]refresh[\'\"]', content, re.IGNORECASE))
        result["has_js_redirect"] = bool(re.search(r'window\.location|location\.href\s*=', content, re.IGNORECASE))

    return result

def scan_directory(path):
    report = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".html"):
                full_path = os.path.join(root, file)
                report.append(check_html_file(full_path))
    return report

def write_report(results):
    with open("检测报告.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"📄 {r['filename']}\n")
            f.write(f"  title标签：{'✅' if r['has_title'] else '❌ 缺失'}\n")
            f.write(f"  description标签：{'✅' if r['has_description'] else '❌ 缺失'}\n")
            f.write(f"  alt属性：{'✅' if r['has_alt'] else '❌ 缺失'}\n")
            f.write(f"  内容长度：{r['content_length']} 字符{' ✅' if r['content_length'] >= 300 else ' ❌ 太短'}\n")
            if r['has_meta_refresh']:
                f.write("  ⚠ 存在 meta refresh 跳转\n")
            if r['has_js_redirect']:
                f.write("  ⚠ 存在 JavaScript 跳转\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    result = scan_directory(TARGET_DIR)
    write_report(result)
    print("✅ 检查完成，已生成《检测报告.txt》")
