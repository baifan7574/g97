# enrich_keywords.py
# 读取 ./keywords/*.txt -> 调用 Google Trends(免费) + Keywords Everywhere API(可选)
# 输出到 ./keywords_enriched/<category>.csv

import os, csv, time, json, math
from pathlib import Path

# ---- 可选：Keywords Everywhere API 配置（有 Key 才会生效） ----
KE_API_KEY = os.environ.get("KE_API_KEY", "").strip()  # 在 PowerShell: $env:KE_API_KEY="你的KEY"
KE_ENDPOINT = "https://api.keywordseverywhere.com/v1/get_keyword_data"
KE_COUNTRY  = "us"    # 可改：us/gb/ca/au/de/fr/es/it/jp...
KE_CURRENCY = "usd"
KE_CHUNK    = 100     # KE 一次最多 100 词，自动分批
KE_SLEEP    = 1.2     # 每批之间等待，防止配额抖动

ROOT = Path(".")
IN_DIR  = ROOT / "keywords"
OUT_DIR = ROOT / "keywords_enriched"
OUT_DIR.mkdir(exist_ok=True)

# ---- Google Trends (pytrends) ----
# pip install pytrends requests
try:
    from pytrends.request import TrendReq
    PYTRENDS_OK = True
except Exception:
    PYTRENDS_OK = False

def trends_scores(keywords, geo="US", timeframe="today 12-m"):  # 返回 {kw: score_0_100}
    """对 keywords 批量打热度分。Trends 每次最多 5 个关键词，取最近12个月的平均值（也可取最近点）。"""
    if not PYTRENDS_OK or not keywords:
        return {k: None for k in keywords}
    pt = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.3)
    out = {}
    BATCH = 5
    for i in range(0, len(keywords), BATCH):
        chunk = keywords[i:i+BATCH]
        try:
            pt.build_payload(chunk, timeframe=timeframe, geo=geo)
            df = pt.interest_over_time()
            if df is None or df.empty:  # 可能冷门
                for k in chunk: out[k] = 0
            else:
                # 取近12个月平均值；也可以改成 df.iloc[-1] 取最新点
                for k in chunk:
                    if k in df.columns:
                        val = float(df[k].mean() if not math.isnan(df[k].mean()) else 0)
                        # normalize 到 0-100
                        out[k] = round(min(100, max(0, val)), 1)
                    else:
                        out[k] = 0
        except Exception:
            for k in chunk: out[k] = 0
        time.sleep(0.4)
    return out

def ke_lookup_batch(keywords):
    """Keywords Everywhere API：返回 {kw: {volume, cpc, competition}}。没有 KE_API_KEY 就全 None。"""
    if not KE_API_KEY:
        return {k: {"volume": None, "cpc": None, "competition": None} for k in keywords}
    import requests
    headers = {"User-Agent":"Mozilla/5.0", "Accept":"application/json", "Content-Type":"application/json",
               "Authorization": KE_API_KEY}
    payload = {
        "country": KE_COUNTRY,
        "currency": KE_CURRENCY,
        "dataSource": "gkp",     # Google Keyword Planner
        "kw": keywords
    }
    try:
        r = requests.post(KE_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=12)
        r.raise_for_status()
        data = r.json()
        res = {}
        # 文档结构：{"data":[{"keyword": "...", "volume":1234, "cpc":0.53, "competition":0.29}, ...]}
        for item in data.get("data", []):
            kw = item.get("keyword", "")
            res[kw] = {
                "volume": item.get("volume"),
                "cpc": item.get("cpc"),
                "competition": item.get("competition")
            }
        # 不在返回里的关键词补 None
        for k in keywords:
            if k not in res:
                res[k] = {"volume": None, "cpc": None, "competition": None}
        return res
    except Exception:
        return {k: {"volume": None, "cpc": None, "competition": None} for k in keywords}

def ke_lookup_all(keywords):
    """自动分批 KE 查询，合并结果。"""
    if not KE_API_KEY:
        return {k: {"volume": None, "cpc": None, "competition": None} for k in keywords}
    out = {}
    for i in range(0, len(keywords), KE_CHUNK):
        chunk = keywords[i:i+KE_CHUNK]
        res = ke_lookup_batch(chunk)
        out.update(res)
        time.sleep(KE_SLEEP)
    return out

def read_keywords(file):
    arr = [x.strip() for x in file.read_text("utf-8", errors="ignore").splitlines() if x.strip()]
    # 去重小写
    seen=set(); out=[]
    for s in arr:
        k = " ".join(s.split()).lower()
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out

def process_one(category, path):
    kws = read_keywords(path)
    if not kws:
        print(f"[SKIP] {category}: 无关键词")
        return
    print(f"[RUN] {category}: {len(kws)} 个关键词")

    # 1) Google Trends 分数
    tmap = trends_scores(kws)

    # 2) KE 搜索量（可选）
    kmap = ke_lookup_all(kws)

    # 3) 写 CSV
    out_csv = OUT_DIR / f"{category}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["keyword","trend_score","volume","cpc","competition"])
        for kw in kws:
            ts = tmap.get(kw)
            km = kmap.get(kw, {})
            w.writerow([kw, ts, km.get("volume"), km.get("cpc"), km.get("competition")])
    print(f"[OK]  写出 {out_csv}")

def main():
    if not IN_DIR.exists():
        print("❌ 未找到 ./keywords 目录"); return
    files = sorted(IN_DIR.glob("*.txt"))
    if not files:
        print("❌ ./keywords 为空，请先生成关键词"); return
    for fp in files:
        process_one(fp.stem, fp)
    print("✅ 全部完成")

if __name__ == "__main__":
    main()
