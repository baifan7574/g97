# -*- coding: utf-8 -*-
"""
auto2_generate_fixed_loop_autopath.py
目的：保持与你现有 BAT 完全兼容；只需双击原来的 bat。
做法：自动扫描当前目录的 config_*.json，逐分类批量出图（9:16写实、干净无字、随机不重复）。
依赖：AUTOMATIC1111 启动时加 --api；（可选）ADetailer 插件。
"""
import os, json, time, base64, datetime, random
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
SERVER = os.environ.get("SD_SERVER", "http://127.0.0.1:7860")

def load_configs():
    return sorted(HERE.glob("config_*.json"))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def choose_one(lst):
    return random.choice(lst) if lst else ""

def build_prompt(cfg):
    base = cfg["base_style"]
    parts = []
    for key in ["scene","outfit","lighting","color","pose","camera"]:
        items = cfg.get("prompt_blocks",{}).get(key, [])
        if items: parts.append(choose_one(items))
    parts = [p for p in parts if p]
    uniq = []
    for p in parts:
        if p not in uniq: uniq.append(p)
    return f"{base}, " + ", ".join(uniq)

def seed_for(cfg, i):
    mode = cfg.get("seed_mode","site")     # 默认按站点区间
    if mode == "fixed":
        s = int(cfg.get("seed_fixed", 123456))
    elif mode == "site":
        site_id = int(cfg.get("site_id", 0))
        s = random.randint(site_id*100000, site_id*100000+99999)
    else:
        s = -1
    if s >= 0: s = s + i                    # 轻抖动避免重复
    return s

def save_image(b64, outdir: Path, idx: int):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{idx:02d}.jpg"
    ensure_dir(outdir)
    (outdir/fname).write_bytes(base64.b64decode(b64))
    return str(outdir/fname)

def call_txt2img(payload):
    url = SERVER.rstrip("/") + "/sdapi/v1/txt2img"
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()

def try_with_or_without_adetailer(payload):
    try:
        return call_txt2img(payload)
    except Exception:
        p2 = dict(payload); p2.pop("alwayson_scripts", None)
        return call_txt2img(p2)

def build_payload(cfg, prompt, negative, seed):
    p = {
        "prompt": prompt,
        "negative_prompt": negative,
        "sampler_name": cfg["sampler_name"],
        "steps": int(cfg["steps"]),
        "cfg_scale": float(cfg["cfg_scale"]),
        "width": int(cfg["width"]),
        "height": int(cfg["height"]),
        "seed": seed,
        "restore_faces": False,
        "enable_hr": bool(cfg["hires_fix"]["enable"]),
        "hr_scale": float(cfg["hires_fix"]["scale"]),
        "denoising_strength": float(cfg["hires_fix"]["denoise"]),
        "hr_upscaler": cfg["hires_fix"]["upscaler"],
        "hr_second_pass_steps": 0,
    }
    if cfg.get("adetailer",{}).get("enable", False):
        ad = cfg["adetailer"]
        p["alwayson_scripts"] = {"ADetailer": {
            "args": [
                {"ad_model":"face_yolov8n.pt",
                 "ad_prompt": ad.get("face_prompt","clear pupils, sharp eyelashes, well-defined lips, natural skin texture"),
                 "ad_denoising_strength": float(ad.get("face_denoise",0.38)),
                 "ad_confidence": float(ad.get("confidence",0.3))},
                {"ad_model":"hand_yolov8n.pt",
                 "ad_prompt": ad.get("hand_prompt","well-formed hands, natural fingers"),
                 "ad_denoising_strength": float(ad.get("hand_denoise",0.30)),
                 "ad_confidence": float(ad.get("confidence",0.3))}
            ]
        }}
    return p

def main():
    print("🔁 扫描分类配置并开始批量生成…")
    files = load_configs()
    if not files:
        print("未发现 config_*.json，退出。"); return
    print("已检测：", ", ".join([f.name for f in files]))

    for cfg_path in files:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        outdir = Path(cfg.get("outdir", f"./output/{cfg.get('category','misc')}")).resolve()
        n = int(cfg.get("images", 20)); used=set(); ok=0
        print(f"\n=== 分类 {cfg.get('category')} | 目标 {n} 张 | 输出 {outdir} ===")
        for i in range(1, n+1):
            for _ in range(30):
                prompt = build_prompt(cfg)
                if prompt not in used: used.add(prompt); break
            seed = seed_for(cfg, i)
            payload = build_payload(cfg, prompt, cfg["negative_prompt"], seed)
            try:
                resp = try_with_or_without_adetailer(payload)
                path = save_image(resp["images"][0], outdir, i)
                ok += 1; print(f"[OK] {cfg.get('category')} #{i:02d} -> {path}")
            except Exception as e:
                print(f"[失败] {cfg.get('category')} #{i:02d}：{e}")
            time.sleep(0.2)
        print(f"完成 {cfg.get('category')}：{ok}/{n} 张")

if __name__ == "__main__":
    main()
