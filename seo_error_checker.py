# seo_error_checker.py  —— 支持单站 (--root) 或 sites.txt 两种模式
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
import argparse

def check_files(html_files, log_path):
    issue_counter = Counter()
    total_checked = 0
    with open(log_path, "w", encoding="utf-8") as log_file:
        for i, file in enumerate(html_files, 1):
            try:
                html = file.read_text(encoding="utf-8", errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                issues = []

                if not soup.title or not (soup.title.string or "").strip():
                    issues.append("缺少 <title>")
                if not soup.find("meta", {"name": "description"}):
                    issues.append("缺少 <meta description>")
                if any(not img.get("alt") for img in soup.find_all("img")):
                    issues.append("缺少 img alt")
                if soup.find("meta", {"name": "robots", "content": "noindex"}):
                    issues.append("含 noindex 标签")

                canonicals = soup.find_all("link", {"rel": "canonical"})
                if len(canonicals) == 0:
                    issues.append("缺少 canonical")
                elif len(canonicals) > 1:
                    issues.append("多个 canonical 冲突")

                if len(soup.get_text().strip()) < 100:
                    issues.append("内容过少 thin content")

                if issues:
                    log_file.write(f"[{file}] \n")
                    for issue in issues:
                        log_file.write(f"  - {issue}\n")
                        issue_counter[issue] += 1
                    log_file.write("\n")
                total_checked += 1
                if i % 50 == 0:
                    print(f"[进度] 已检查 {i}/{len(html_files)} 个页面")
            except Exception as e:
                log_file.write(f"[ERROR] {file}: {e}\n")

        log_file.write("\n=== 总结统计 ===\n")
        log_file.write(f"共检查 {total_checked} 个页面\n")
        for issue, count in issue_counter.most_common():
            log_file.write(f"{issue}: {count} 个页面\n")

    print(f"[完成] 共检查 {total_checked} 个页面 ✅，结果见 {log_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="要检查的单站根目录（传了就只检查这个目录）")
    ap.add_argument("--base", default="D:/项目/", help="多站模式的根目录（默认 D:/项目/）")
    ap.add_argument("--sites", default="sites.txt", help="多站模式下的站点列表文件")
    args = ap.parse_args()

    if args.root:
        root = Path(args.root).resolve()
        html_files = list(root.rglob("*.html"))
        print(f"[INFO] 单站模式：{root}，共收集 {len(html_files)} 个 HTML")
        check_files(html_files, root / "seo_error_log.txt")
    else:
        base_path = Path(args.base)
        sites_file = base_path / args.sites
        targets = []
        if sites_file.exists():
            targets = [line.strip() for line in sites_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        html_files = []
        for folder in targets:
            p = base_path / folder
            if p.exists():
                html_files.extend(p.rglob("*.html"))
        print(f"[INFO] 多站模式：从 {len(targets)} 个目录收集到 {len(html_files)} 个 HTML")
        check_files(html_files, Path("seo_error_log.txt"))

if __name__ == "__main__":
    main()
