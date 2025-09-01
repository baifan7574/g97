# -*- coding: utf-8 -*-
"""
NorthBeam Ads Injector v1.3 SAFE
- 首页：顶部/底部横幅、左右竖条、悬浮、弹窗
- 其他页：顶部/底部横幅、悬浮
- 所有页：<head> 自动插入手机端广告
- 不破坏结构：固定定位 + body padding；带注入标记，幂等
- 仅“本网站”判定的首页文件视作首页（见 HOME_CANDIDATES）
"""

import json, re, shutil, time
from pathlib import Path

# ============ 只改这一行：当前要处理的网站目录（例如 g40 / g50） ============
ACTIVE_SITE = Path(r".").resolve()
# ======================================================================

CONFIG_FILE = Path("ads_mapping.json")  # 配置文件与脚本同级
SENTINEL_NAME = ".enable_ads"           # 哨兵文件（根目录需有此空文件才允许写入）
DRY_RUN = False                         # True=演练不写盘；False=正式写入
BACKUP_BEFORE_WRITE =False             # 写入前备份
HOME_CANDIDATES = ["index.html", "index.htm", "index_template.html", "home.html"]  # 首页候选

MARKERS = {
    "css": "<!-- NB:ADS-CSS -->",
    "top": "<!-- NB:AD-Top -->",
    "bottom": "<!-- NB:AD-Bottom -->",
    "left": "<!-- NB:AD-LeftSky -->",
    "right": "<!-- NB:AD-RightSky -->",
    "float": "<!-- NB:AD-Floating -->",
    "popup": "<!-- NB:AD-Popup -->",
    "mobile_head": "<!-- NB:AD-MobileHead -->"
}

# 样式块（重要：不使用 .format()，避免 {} 触发格式化错误）
CSS_BLOCK = MARKERS["css"] + """
<style id="nb-ads-css">
@media (min-width: 1024px){
  body.nb-has-top { padding-top: 96px; }
  body.nb-has-bottom { padding-bottom: 96px; }
}
.nb-topbar-wrap,.nb-bottombar-wrap{
  position: fixed; left:0; right:0; z-index: 2147483000;
  display:flex; justify-content:center; pointer-events:auto;
}
.nb-topbar-wrap{ top:0; }
.nb-bottombar-wrap{ bottom:0; }
.nb-stick{ position: fixed; top: 100px; z-index: 2147483000; }
.nb-left{ left: 8px; }
.nb-right{ right: 8px; }
.nb-float{
  position: fixed; right: 12px; bottom: 12px; z-index: 2147483000;
  box-shadow: 0 6px 24px rgba(0,0,0,.25); border-radius: 8px; background: rgba(0,0,0,0);
}
.nb-modal{ position: fixed; inset:0; background: rgba(0,0,0,.55); display:flex; align-items:center; justify-content:center; z-index:2147483000; }
.nb-modal-inner{ position: relative; background:#111; padding:10px; border-radius:10px; max-width:92vw; max-height:85vh; overflow:auto; }
.nb-close{ position:absolute; right:8px; top:6px; font-size:24px; background:#000; color:#fff; border:0; width:36px; height:36px; border-radius:50%; cursor:pointer; }
@media (max-width: 767px){
  body.nb-has-top { padding-top: 56px; }
  body.nb-has-bottom { padding-bottom: 56px; }
}
</style>
<script>(function(){if(document.documentElement.dataset.nbAdsCss)return;document.documentElement.dataset.nbAdsCss="1";})();</script>
"""

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="ignore")

def write_text(p: Path, s: str, backup_dir: Path | None):
    if DRY_RUN:
        print(f"[DRY] would write: {p}")
        return
    if BACKUP_BEFORE_WRITE and backup_dir is not None:
        backup_path = backup_dir / p.relative_to(ACTIVE_SITE)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, backup_path)
    p.write_text(s, encoding="utf-8")

def ensure_head(html: str) -> str:
    if "</head>" not in html.lower():
        html = re.sub("(?i)<html[^>]*>", "\\g<0>\n<head></head>", html, count=1)
    return html

def insert_into_head_once(html: str, block: str, marker: str) -> str:
    if marker.lower() in html.lower(): return html
    html = ensure_head(html)
    return re.sub("(?i)</head>", block + "\n</head>", html, count=1)

def insert_after_body_open(html: str, block: str, marker: str, add_body_class: str = "") -> str:
    if marker.lower() in html.lower(): return html
    def add_class(m):
        tag = m.group(0)
        if re.search(r'(?i)class\s*=', tag):
            return re.sub(r'(?i)class\s*=\s*([\'"])(.*?)\1',
                          lambda mm: f'class="{mm.group(2)} {add_body_class}"', tag, count=1)
        else:
            return tag.replace("<body", f"<body class=\"{add_body_class}\"", 1)
    html2, n = re.subn(r'(?i)<body[^>]*>', add_class, html, count=1)
    if n == 0:
        html2 = "<body class=\"%s\">" % add_body_class + html + "</body>"
    html2 = re.sub(r'(?i)<body[^>]*>', lambda m: m.group(0) + "\n" + marker + "\n" + block, html2, count=1)
    return html2

def insert_before_body_close(html: str, block: str, marker: str, add_body_class: str = "") -> str:
    if marker.lower() in html.lower(): return html
    if add_body_class:
        html = re.sub(r'(?i)<body([^>]*)class\s*=\s*([\'"])(.*?)\2',
                      lambda m: f"<body{m.group(1)}class=\"{m.group(3)} {add_body_class}\"", html, count=1)
        if re.search(r'(?i)<body[^>]*class=', html) is None:
            html = re.sub(r'(?i)<body', f'<body class="{add_body_class}"', html, count=1)
    return re.sub("(?i)</body>", marker + "\n" + block + "\n</body>", html, count=1)

# 修复：不在 wrapper 里重复 marker（避免双 marker）
def insert_fixed_slot(html: str, block: str, marker: str, side: str) -> str:
    if marker.lower() in html.lower(): return html
    wrapper = f'<div class="nb-stick nb-{side}" data-nb="{side}">{block}</div>'
    return insert_after_body_open(html, wrapper, marker)

def insert_floating(html: str, block: str, marker: str, delay_ms: int) -> str:
    if marker.lower() in html.lower(): return html
    wrapper = f"""
<div id="nb-float" class="nb-float" style="display:none">{block}</div>
<script>setTimeout(function(){{var a=document.getElementById('nb-float');if(a)a.style.display='block';}}, {delay_ms});</script>
"""
    return insert_before_body_close(html, wrapper, marker)

def insert_popup(html: str, block: str, marker: str) -> str:
    if marker.lower() in html.lower(): return html
    return insert_before_body_close(html, block, marker)

def choose(snippets): return snippets[0] if snippets else ""

def assert_config_safe(raw: str):
    # 防止配置被双重转义，导致浏览器不执行
    if r"\\\"" in raw or r"<\\/" in raw:
        raise ValueError("ads_mapping.json 含有转义片段（\\\" 或 <\\/script>）。请使用干净版配置后再运行。")

def detect_home(active_site: Path) -> Path | None:
    for name in HOME_CANDIDATES:
        p = active_site / name
        if p.exists(): return p.resolve()
    return None

def main():
    print(f"[INFO] ACTIVE_SITE = {ACTIVE_SITE}")
    if not ACTIVE_SITE.exists():
        raise SystemExit(f"[ABORT] 站点不存在：{ACTIVE_SITE}")
    if not (ACTIVE_SITE / SENTINEL_NAME).exists():
        raise SystemExit(f"[ABORT] 缺少哨兵文件 {SENTINEL_NAME} ，已拒绝修改。请在站点根目录创建一个空文件再运行。")

    home_path = detect_home(ACTIVE_SITE)
    if home_path is None:
        print(f"[WARN] 未发现首页文件（候选：{', '.join(HOME_CANDIDATES)}），将不会注入首页专属位。")
    else:
        print(f"[INFO] 检测到首页：{home_path}")

    raw_cfg_text = CONFIG_FILE.read_text(encoding="utf-8")
    assert_config_safe(raw_cfg_text)
    cfg = json.loads(raw_cfg_text)

    popup_delay = int(cfg.get("global", {}).get("popup_delay_ms", 1500))
    float_delay = int(cfg.get("global", {}).get("float_show_delay_ms", 800))
    enable_home = bool(cfg.get("global", {}).get("enable_on_home", True))
    enable_inner = bool(cfg.get("global", {}).get("enable_on_inner", True))

    if "home" in cfg and "popup" in cfg["home"]:
        cfg["home"]["popup"] = [s.replace("{{POPUP_DELAY}}", str(popup_delay)) for s in cfg["home"]["popup"]]

    backup_dir = None
    if BACKUP_BEFORE_WRITE and not DRY_RUN:
        backup_dir = ACTIVE_SITE / ".nbbackup" / time.strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 将备份原文件到：{backup_dir}")

    files = list(ACTIVE_SITE.rglob("*.html"))
    for fp in files:
        rel = fp.relative_to(ACTIVE_SITE).as_posix()
        html = read_text(fp)

        is_home = (home_path is not None and fp.resolve() == home_path)
        role = "home" if is_home else "inner"
        enabled = (enable_home and is_home) or (enable_inner and not is_home)
        if not enabled:
            print(f"[SKIP] {rel}（该角色未启用）")
            continue

        # 1) 全站 CSS
        if cfg.get("global", {}).get("css_once", True):
            html = insert_into_head_once(html, CSS_BLOCK, MARKERS["css"])
        # 2) 移动端/全站 <head>
        for snip in cfg.get("global", {}).get("mobile_head_snippets", []):
            html = insert_into_head_once(html, MARKERS["mobile_head"] + "\n" + snip, MARKERS["mobile_head"])
        # 3) 顶部横幅
        if "top_banner" in cfg.get(role, {}):
            block = f'<div class="nb-topbar-wrap"><div class="nb-ad">{choose(cfg[role]["top_banner"])}</div></div>'
            html = insert_after_body_open(html, block, MARKERS["top"], add_body_class="nb-has-top")
        # 4) 底部横幅
        if "bottom_banner" in cfg.get(role, {}):
            block = f'<div class="nb-bottombar-wrap"><div class="nb-ad">{choose(cfg[role]["bottom_banner"])}</div></div>'
            html = insert_before_body_close(html, block, MARKERS["bottom"], add_body_class="nb-has-bottom")
        # 5) 左右竖条（仅首页）
        if is_home:
            if "left_sky" in cfg.get("home", {}):
                html = insert_fixed_slot(html, choose(cfg["home"]["left_sky"]), MARKERS["left"], "left")
            if "right_sky" in cfg.get("home", {}):
                html = insert_fixed_slot(html, choose(cfg["home"]["right_sky"]), MARKERS["right"], "right")
        # 6) 悬浮
        if "floating" in cfg.get(role, {}):
            html = insert_floating(html, choose(cfg[role]["floating"]), MARKERS["float"], float_delay)
        # 7) 弹窗（首页）
        if is_home and "popup" in cfg.get("home", {}):
            html = insert_popup(html, choose(cfg["home"]["popup"]), MARKERS["popup"])

        write_text(fp, html, backup_dir)
        print(f"[OK] Injected: {rel} ({'HOME' if is_home else 'INNER'})")

    print(f"Done. {'DRY-RUN（未写盘）完成' if DRY_RUN else '已写入修改'} → {ACTIVE_SITE}")

if __name__ == "__main__":
    main()
