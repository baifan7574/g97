# select_keywords.py — 站群级“已用词”补丁版
from pathlib import Path
import csv, re, os

ROOT = Path(".")
IN_TXT  = ROOT / "keywords"
IN_CSV  = ROOT / "keywords_enriched"
OUT_DIR = ROOT / "selected_keywords"
OUT_DIR.mkdir(exist_ok=True)

# ===== 站群级“已用词”位置（很关键）=====
# 1) 推荐：改成你的绝对路径，实现跨站共享，不撞词
#    例如：USED_GLOBAL_PATH = r"D:\项目\used_keywords_global.txt"
# 2) 不想跨站共享就保持默认（当前项目根目录）
USED_GLOBAL_PATH = os.environ.get("NB_USED_GLOBAL", "").strip() or str((ROOT / "used_keywords_global.txt").resolve())

# —— 阈值（保持与你现有一致，略放宽）——
MIN_VOLUME     = 10
MAX_VOLUME     = 10000
MAX_COMP       = 0.85
MIN_TREND      = 0
TOP_N_PER_CAT  = 800
REQUIRE_ENGLISH= False
GLOBAL_BAN = {"wallpaper","download","emoji","gif","porn","sex","xxx","nude"}
MIN_LEN   = 4
MIN_WORDS = 2

def is_cn(s:str)->bool:
    return re.search(r"[\u4e00-\u9fff]", s) is not None

def ok_kw(k:str)->bool:
    k = re.sub(r"\s+"," ", k.strip().lower())
    if not k: return False
    if REQUIRE_ENGLISH and is_cn(k): return False
    if any(b in k for b in GLOBAL_BAN): return False
    if len(k) < MIN_LEN: return False
    if len(k.split()) < MIN_WORDS: return False
    return True

def load_used()->set:
    p = Path(USED_GLOBAL_PATH)
    if p.exists():
        return {re.sub(r"\s+"," ",x.strip().lower())
                for x in p.read_text("utf-8",errors="ignore").splitlines() if x.strip()}
    return set()

def append_used(lst):
    if not lst: return
    p = Path(USED_GLOBAL_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for k in lst: f.write(k+"\n")

def pick_from_csv(fp:Path):
    rows=[]
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        r=csv.DictReader(f)
        for row in r:
            kw = re.sub(r"\s+"," ", (row.get("keyword") or "").strip().lower())
            if not ok_kw(kw): continue
            def fnum(x):
                try:
                    if x is None or str(x).strip()=="": return None
                    return float(x)
                except: return None
            trend = fnum(row.get("trend_score"))
            vol   = fnum(row.get("volume"))
            comp  = fnum(row.get("competition"))
            rows.append((kw, trend, vol, comp))

    total = len(rows)
    good=[]
    for kw,trend,vol,comp in rows:
        if vol is not None and comp is not None:
            if (vol >= MIN_VOLUME) and (vol <= MAX_VOLUME) and (comp <= MAX_COMP):
                good.append((kw, vol, comp, trend if trend is not None else -1))
        else:
            if (trend or 0) >= MIN_TREND:
                good.append((kw, -1,  9, trend if trend is not None else 0))

    good.sort(key=lambda x: ((x[1] if x[1] is not None else -1),
                             (x[3] if x[3] is not None else -1),
                             -x[2]), reverse=True)

    if not good and total>0:
        good = [(kw, -1, 9, trend if trend is not None else 0) for (kw,trend,vol,comp) in rows]

    seen=set(); out=[]
    for kw,_,_,_ in good:
        if kw in seen: continue
        seen.add(kw); out.append(kw)
        if len(out)>=TOP_N_PER_CAT: break

    print(f"    [CSV] 读取{total}条，筛后{len(out)}条（若0则走兜底）")
    return out

def pick_from_txt(fp:Path):
    arr=[re.sub(r"\s+"," ",x.strip().lower())
         for x in fp.read_text("utf-8",errors="ignore").splitlines() if x.strip()]
    total=len(arr)
    kept=[]
    seen=set()
    for kw in arr:
        if not ok_kw(kw): continue
        if kw in seen: continue
        seen.add(kw); kept.append(kw)
        if len(kept)>=TOP_N_PER_CAT: break
    if not kept and total>0:
        kept = arr[:TOP_N_PER_CAT]
    print(f"    [TXT] 读取{total}条，筛后{len(kept)}条（若0则走兜底）")
    return kept

def main():
    cats=set([p.stem for p in IN_TXT.glob("*.txt")]) | set([p.stem for p in IN_CSV.glob("*.csv")])
    if not cats:
        print("❌ 未发现 keywords/ 或 keywords_enriched/ 下的分类文件"); return
    print(f"[INFO] 分类：{len(cats)} 个")
    used = load_used()

    for cat in sorted(cats):
        csv_fp = IN_CSV / f"{cat}.csv"
        txt_fp = IN_TXT / f"{cat}.txt"

        print(f"[RUN] {cat}")
        if csv_fp.exists():
            picked = pick_from_csv(csv_fp)
        elif txt_fp.exists():
            picked = pick_from_txt(txt_fp)
        else:
            print("    [WARN] 无 CSV/TXT，跳过"); picked = []

        final=[]
        for kw in picked:
            if kw in used: continue
            used.add(kw); final.append(kw)

        out = OUT_DIR / f"{cat}.txt"
        out.write_text("\n".join(final), "utf-8")
        print(f"[OK]  {cat}: 写出 {len(final)} 条 -> {out}\n")

        append_used(final)

    print(f"✅ 完成：输出 {OUT_DIR}\\*.txt ；全局去重文件：{USED_GLOBAL_PATH}")

if __name__ == "__main__":
    main()
