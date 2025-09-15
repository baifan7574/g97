# -*- coding: utf-8 -*-
"""
kw_persist_and_fill.py
功能：
1) 为每个页面生成/复用 1 个稳定关键词，并持久化到 .kw_map.json（url->keyword）。
2) 自动为分类页/单图页插入 80-200 词正文段落（带标记，二次运行可覆盖更新，不重复叠加）。
3) 如首张 <img> 缺失 alt，则用该关键词补上（仅在 alt 为空时才写，避免与其他脚本冲突）。

使用：
python kw_persist_and_fill.py --root . --pool keywords\\selected.txt --min-words 100 --max-words 180
参数说明见 main() 下方 argparse。
"""
import os, re, json, random, hashlib, argparse

# ----------- 可按需忽略的目录/文件 -----------
SKIP_DIRS = {'.git', 'assets', 'static', 'vendor', 'node_modules', '.venv', 'venv'}
HTML_EXTS = {'.html', '.htm'}

MAP_FILE = '.kw_map.json'   # 持久化映射：{"rel/url.html": "keyword"}
DESC_START = '<!--AUTO_DESC_START-->'
DESC_END   = '<!--AUTO_DESC_END-->'

def is_html(path):
    return os.path.splitext(path)[1].lower() in HTML_EXTS

def rel_url(root, path):
    rel = os.path.relpath(path, root).replace('\\','/')
    return rel

def load_kw_map(root):
    p = os.path.join(root, MAP_FILE)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_kw_map(root, mp):
    p = os.path.join(root, MAP_FILE)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(mp, f, ensure_ascii=False, indent=2)

def load_pool(pool_file):
    if pool_file and os.path.exists(pool_file):
        with open(pool_file, 'r', encoding='utf-8') as f:
            words = [ln.strip() for ln in f if ln.strip()]
        return list(dict.fromkeys(words))  # 去重但保持顺序
    return []

def load_global_used(global_path):
    if not global_path: return set()
    if os.path.exists(global_path):
        with open(global_path, 'r', encoding='utf-8') as f:
            return set(ln.strip() for ln in f if ln.strip())
    return set()

def append_global_used(global_path, keyword):
    if not global_path or not keyword: return
    os.makedirs(os.path.dirname(global_path), exist_ok=True)
    with open(global_path, 'a', encoding='utf-8') as f:
        f.write(keyword.strip() + '\n')

def pick_keyword(url, kw_map, pool, used_global):
    """为 url 挑一个关键词：优先复用 kw_map；否则从 pool 里按顺序挑一个未被 used_global 使用过的。"""
    if url in kw_map and kw_map[url]:
        return kw_map[url], False  # False = 不是新分配
    for kw in pool:
        if kw not in used_global and kw not in kw_map.values():
            kw_map[url] = kw
            return kw, True
    # 如果池子空了，就退而求其次：用文件名派生一个关键词，避免空
    fallback = os.path.splitext(os.path.basename(url))[0].replace('_',' ').replace('-',' ')
    if not fallback:
        fallback = 'photo gallery'
    kw_map[url] = fallback
    return fallback, True

def detect_page_type(url):
    """
    简单判断：/xxx/index.html 或路径含 /category/ /categories/ 视为分类页；其余视为单图页
    你可以按你站点规则微调。
    """
    u = url.lower()
    if u.endswith('/index.html') or '/category/' in u or '/categories/' in u:
        return 'category'
    return 'image'

def seeded_random_text(seed_str, keyword, ptype, min_words=100, max_words=180):
    """
    用稳定随机（基于 url 的 hash 做种子）生成 100~180 词之间的段落，包含 keyword。
    这样同一 url 每次生成的文本一致；不同 url 则不同。
    """
    seed = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32)
    rnd = random.Random(seed)

    # 几个模板片段池（你可以按站点风格扩充）
    intros = [
        f"This page explores {keyword} with a practical focus on visual detail and browsing experience.",
        f"Here we highlight {keyword}, aiming for clean structure, quick scanning, and useful context.",
        f"Designed for readers looking into {keyword}, this page emphasizes clarity and consistency."
    ]
    bodies_cat = [
        "You will find a concise introduction to the theme, suggestions for discovery, and guidance to navigate related sections. The layout balances thumbnails and text so each visit feels lightweight but informative. Frequent updates keep the collection fresh.",
        "We group similar items to reduce repetition while preserving variety. This structure helps search engines understand relationships and helps visitors jump between related pages with minimal friction.",
        "Short descriptive blurbs add context to images, improving accessibility and search relevance without overwhelming the visuals. Internal links further tie categories together."
    ]
    bodies_img = [
        "The image aims to deliver a straightforward visual impression while keeping the file lightweight. A brief explanation clarifies the subject and lighting so visitors can quickly decide where to go next.",
        "Alt text and headings are optimized to make the content accessible and to provide consistent cues across the site. Subtle differences in wording help avoid duplication across similar pages.",
        "Internal navigation leads to related items with comparable tone or composition. This reduces bounce and supports exploration within the same theme."
    ]
    closings = [
        "If you are comparing alternatives, keep an eye on subtle differences in framing, contrast, and color balance.",
        "For more context, browse related entries linked nearby; each page offers a slightly different angle to limit overlap.",
        "Bookmark the page if it’s useful; updates aim to improve clarity, speed, and overall structure over time."
    ]

    targets = rnd.randint(min_words, max_words)
    parts = [rnd.choice(intros)]
    pool = bodies_cat if ptype == 'category' else bodies_img
    while len(" ".join(parts).split()) < targets:
        parts.append(rnd.choice(pool))
        if rnd.random() < 0.5:
            parts.append(rnd.choice(closings))
    text = " ".join(parts)

    # 保证 keyword 至少出现一次
    if keyword.lower() not in text.lower():
        text = f"{keyword}. " + text
    return text

def inject_auto_desc(html, desc_html):
    """
    在 HTML 中插入/更新自动描述块：
    - 若存在 <!--AUTO_DESC_START-->...<!--AUTO_DESC_END--> 则替换；
    - 否则优先插到 </main> 前；没有 <main> 就插到 </body> 前。
    """
    if DESC_START in html and DESC_END in html:
        pattern = re.compile(re.escape(DESC_START) + r'.*?' + re.escape(DESC_END), re.S | re.I)
        return pattern.sub(DESC_START + '\n' + desc_html + '\n' + DESC_END, html)

    block = f"""{DESC_START}
<section class="auto-desc" style="max-width:900px;margin:1rem auto;line-height:1.6;">
  {desc_html}
</section>
{DESC_END}"""

    # 优先放在 </main> 之前
    m = re.search(r'</main\s*>', html, flags=re.I)
    if m:
        idx = m.start()
        return html[:idx] + block + html[idx:]
    # 其次放在 </body> 之前
    m = re.search(r'</body\s*>', html, flags=re.I)
    if m:
        idx = m.start()
        return html[:idx] + block + html[idx:]
    # 都没有就直接追加
    return html + "\n" + block + "\n"

def ensure_first_img_alt(html, keyword):
    """
    如果首张 <img> 没有 alt，则补上；若已有 alt，不改（避免与其他脚本冲突）。
    """
    def repl(m):
        tag = m.group(0)
        # 已有 alt 的不处理
        if re.search(r'\salt\s*=\s*["\']', tag, flags=re.I):
            return tag
        # 没有 alt：在最后一个 > 前插入 alt
        tag2 = re.sub(r'>\s*$', f' alt="{keyword}">', tag)
        return tag2

    return re.sub(r'<img\b[^>]*?>', repl, html, count=1, flags=re.I|re.S)

def process_page(root, path, kw_map, pool, used_global, global_path, min_words, max_words):
    url = rel_url(root, path)
    ptype = detect_page_type(url)
    keyword, is_new = pick_keyword(url, kw_map, pool, used_global)

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()

    # 生成稳定描述文本（按 url 作为随机种子，保证每次一致）
    desc_txt = seeded_random_text(url, keyword, ptype, min_words, max_words)
    desc_html = f"<p>{desc_txt}</p>"

    # 注入描述块
    html2 = inject_auto_desc(html, desc_html)

    # 如首图 alt 为空则补上
    html2 = ensure_first_img_alt(html2, keyword)

    if html2 != html:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html2)

    # 新分配的关键词写入全局去重库
    if is_new and global_path:
        append_global_used(global_path, keyword)

    return keyword, is_new

def main():
    ap = argparse.ArgumentParser(description="Persist url→keyword mapping and fill 80–200 word descriptions.")
    ap.add_argument('--root', default='.', help='站点根目录（默认当前目录）')
    ap.add_argument('--pool', default='keywords/selected.txt',
                    help='关键词池文件（每行一个词）。默认 keywords/selected.txt；找不到则自动降级。')
    ap.add_argument('--global-used', default=os.environ.get('NB_USED_GLOBAL', r'D:\project\used_keywords_global.txt'),
                    help='跨站去重词库文件（默认读取环境变量 NB_USED_GLOBAL，否则 D:\\project\\used_keywords_global.txt）')
    ap.add_argument('--min-words', type=int, default=100, help='描述最小词数（默认100）')
    ap.add_argument('--max-words', type=int, default=180, help='描述最大词数（默认180）')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    kw_map = load_kw_map(root)
    pool = load_pool(os.path.join(root, args.pool)) if not os.path.isabs(args.pool) else load_pool(args.pool)
    used_global = load_global_used(args.global_used)

    changed = 0
    assigned_new = 0
    total = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # 跳过部分目录
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if not is_html(fp): continue
            total += 1
            kw, is_new = process_page(root, fp, kw_map, pool, used_global, args.global_used,
                                      args.min_words, args.max_words)
            if is_new:
                assigned_new += 1
            changed += 1

    save_kw_map(root, kw_map)

    # 额外导出 csv 方便你或其他脚本使用
    try:
        out_csv = os.path.join(root, '.kw_map.csv')
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write('url,keyword\n')
            for u, k in kw_map.items():
                f.write(f'{u},{k}\n')
    except Exception:
        pass

    print(f'[OK] processed pages: {total}, changed: {changed}, new_assigned: {assigned_new}')
    print(f'[OK] kw map saved: {os.path.join(root, MAP_FILE)}')
    print(f'[OK] csv exported: {os.path.join(root, ".kw_map.csv")}')
    if args.global_used:
        print(f'[OK] global used file: {args.global_used}')

if __name__ == '__main__':
    main()
