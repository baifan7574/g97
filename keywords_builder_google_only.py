# keywords_builder_google_only.py
# 只用 Google Suggest，实时日志，超时重试，写入 .\keywords\<分类>.txt
import re, json, time
from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(".")
SEEDS = ROOT / "seeds"
OUT   = ROOT / "keywords"

TIMEOUT = 4.0
RETRIES = 2
SLEEP   = (0.6, 1.0)  # 每次请求间隔范围

MIN_LEN = 6
MIN_WORDS = 2
MAX_PER_CAT = 800
GLOBAL_BAN  = {"4k","hd","wallpaper","download","emoji","gif","porn","sex","xxx","nude"}
GLOBAL_KEEP = {"portrait","photo","photography","style","aesthetic","model","girl","woman","female"}

def log(s): print(time.strftime("[%H:%M:%S] "), s, flush=True)
def is_cn(s): return re.search(r"[\u4e00-\u9fff]", s) is not None

def ok_kw(k):
    k = re.sub(r"\s+"," ",k.strip().lower())
    if not k or is_cn(k): return False
    if any(b in k for b in GLOBAL_BAN): return False
    if len(k) < MIN_LEN: return False
    if len(k.split()) < MIN_WORDS: return False
    if GLOBAL_KEEP and not any(g in k for g in GLOBAL_KEEP): return False
    return True

def uniq(lst):
    seen=set(); out=[]
    for s in lst:
        k=re.sub(r"\s+"," ",s.strip().lower())
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out

def http_get(url):
    import random
    for i in range(RETRIES+1):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code == 200:
                return r.text
            log(f"[HTTP] {r.status_code} {url}")
        except Exception as e:
            log(f"[HTTP] {e.__class__.__name__}: {e}")
        time.sleep(min(SLEEP[1], SLEEP[0]*(i+1)))
    return None

def g_suggest(q):
    url = f"https://suggestqueries.google.com/complete/search?client=firefox&q={quote(q)}"
    txt = http_get(url)
    if not txt: return []
    try:
        arr = json.loads(txt)
        return arr[1] if isinstance(arr, list) and len(arr)>1 else []
    except Exception:
        return []

def seeds_of(fp: Path):
    lines=[x.strip() for x in fp.read_text("utf-8",errors="ignore").splitlines() if x.strip()]
    return lines or [fp.stem]

def process_cat(cat, seeds):
    OUT.mkdir(exist_ok=True)
    pool=[]; hit=False
    log(f"====== 分类：{cat}  种子：{seeds} ======")
    for s in seeds:
        base=[s, f"{s} portrait", f"{s} photo", f"{s} photography", f"{s} aesthetic"]
        for q in base:
            log(f"[ASK] {q}")
            sug = g_suggest(q)
            if sug: hit=True
            for w in sug:
                if ok_kw(w): pool.append(w)
            time.sleep(0.7)
    if not hit:
        log("[WARN] Google 无命中：检查网络/种子词是否过于冷门")
    pool = uniq(pool)
    if len(pool) > MAX_PER_CAT: pool = pool[:MAX_PER_CAT]
    out = OUT / f"{cat}.txt"
    out.write_text("\n".join(pool), "utf-8")
    log(f"[WRITE] {out}  数量={len(pool)}")

def main():
    if not SEEDS.exists():
        log("❌ 请在 ./seeds 放入 bedroom.txt / office.txt 等种子文件"); return
    files = sorted(SEEDS.glob("*.txt"))
    if not files:
        log("❌ seeds 目录为空"); return
    for fp in files:
        process_cat(fp.stem, seeds_of(fp))
    log("✅ 完成")

if __name__ == "__main__":
    main()
