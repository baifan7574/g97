import os
import datetime
from bs4 import BeautifulSoup

# ========== 日志初始化 ==========
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

# ========== 核心修复逻辑 ==========
def fix_html_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')

        # 获取 <head> 标签
        head_tag = soup.head
        if head_tag is None:
            log(f"❌ 无法修复：{filepath}（没有<head>标签）")
            return

        # 检查并添加 canonical 标签
        canonical_exists = soup.find("link", {"rel": "canonical"})
        if not canonical_exists:
            canonical_link = soup.new_tag("link", rel="canonical", href=f"./{os.path.basename(filepath)}")
            head_tag.append(canonical_link)
            log(f"🔗 添加 canonical: ./{os.path.basename(filepath)}")

        # 检查并添加结构化 schema（简化版）
        schema_exists = soup.find("script", {"type": "application/ld+json"})
        if not schema_exists:
            schema_tag = soup.new_tag("script", type="application/ld+json")
            schema_tag.string = """{
  "@context": "https://schema.org",
  "@type": "WebPage",
  "name": "Sample Page",
  "description": "Auto SEO Fixer Page"
}"""
            head_tag.append(schema_tag)
            log(f"🧩 添加 schema 标记")

        # 保存修复后的内容
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(soup))

        log(f"✅ 修复完成：{filepath}")

    except Exception as e:
        log(f"❌ 错误处理文件 {filepath}: {e}")

# ========== 资源文件清理 ==========
def delete_empty_or_redirect_html(folder):
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.html'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        html = f.read()
                        if 'window.location.href' in html or len(html.strip()) < 20:
                            os.remove(filepath)
                            log(f"🗑️ 删除失效资源：{file}")
                            count += 1
                except:
                    continue
    return count

# ========== 遍历所有 HTML ==========
def walk_and_fix(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.html'):
                full_path = os.path.join(root, file)
                fix_html_file(full_path)

# ========== 主程序入口 ==========
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"🔧 SEO 自动修复启动：{now}\n")

    current_dir = os.getcwd()

    # 清理无效资源
    removed = delete_empty_or_redirect_html(current_dir)
    log(f"\n✅ 共清理无效资源文件：{removed} 个\n")

    # 开始修复
    walk_and_fix(current_dir)

    # 输出日志文件
    log_path = os.path.join(current_dir, "seo_fixer_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    log(f"\n📄 日志已保存至：{log_path}")
    log("✅ 所有修复任务完成！")
