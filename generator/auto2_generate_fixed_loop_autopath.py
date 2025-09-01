# -*- coding: utf-8 -*-
import os, json, time, random, base64, requests
from datetime import datetime
from typing import Dict, Any, List, Optional

API_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"
TIMEOUT = 180
MAX_RETRY = 3
WAIT_API_SECONDS = 60   # 最长等 API 就绪 60 秒

DEFAULT_CATEGORIES = ["bedroom","dark","office","soft","uniform","shower","mirror","fitness","luxury","redroom"]

def here(): return os.path.dirname(os.path.abspath(__file__))
def parent_here(): return os.path.abspath(os.path.join(here(),".."))

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def wait_api_ready():
    deadline = time.time() + WAIT_API_SECONDS
    while time.time() < deadline:
        try:
            r = requests.get("http://127.0.0.1:7860/sdapi/v1/progress", timeout=5)
            if r.status_code == 200: 
                print("[OK] WebUI API 就绪")
                return
        except: pass
        print("[INFO] 等待 WebUI API 就绪…")
        time.sleep(2)
    print("[WARN] 等待超时，继续尝试请求。")

def load_json(path) -> Optional[Dict[str,Any]]:
    if os.path.isfile(path):
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    return None

def load_keywords(category:str)->List[str]:
    kw_dir = os.path.join(parent_here(),"keywords")
    kw_path = os.path.join(kw_dir,f"{category}.txt")
    if not os.path.isfile(kw_path): return []
    out=[]
    with open(kw_path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            s=line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out

def build_output_dir(category:str)->str:
    out_dir = os.path.abspath(os.path.join(here(),"..",category))
    ensure_dir(out_dir)
    return out_dir

def b64_to_file(b64data:str, out_path:str):
    with open(out_path,"wb") as f:
        f.write(base64.b64decode(b64data))

# —— 默认稳定参数（显存友好 + 画面稳定）——
DEFAULT_PAYLOAD: Dict[str,Any] = {
    "prompt": "",
    "negative_prompt": "(worst quality, low quality, normal quality:1.2), "
                       "(duplicate:1.4), (two faces:1.4), (two heads:1.4), (twins:1.4), "
                       "split screen, collage, multiple reflections, extra head, extra body, "
                       "bad anatomy, missing fingers, bad hands, deformed, watermark, text, logo, blurry",
    "sampler_name": "DPM++ 2M Karras",
    "steps": 30,
    "cfg_scale": 6.5,
    "width": 768,
    "height": 1344,
    "seed": -1,
    "save_images": False,
    "restore_faces": False,
    "enable_hr": True,
    "hr_scale": 1.5,
    "hr_upscaler": "R-ESRGAN 4x+ Anime6B",   # 若你本地叫别的名，改成对应名字即可
    "denoising_strength": 0.30,
    "override_settings": {
        "outdir_txt2img_samples": "",
        "outdir_txt2img_grids": "",
        "outdir_save": ""
    },
    "batch_size": 1
}

ALLOWED_KEYS = {
    "prompt","base_prompt","negative_prompt","sampler_name","steps","cfg_scale",
    "width","height","seed","enable_hr","hr_scale","hr_upscaler","denoising_strength",
    "batch_size","images_count"
}

def merge_payload(base:Dict[str,Any], override:Dict[str,Any])->Dict[str,Any]:
    x = base.copy()
    for k,v in override.items():
        if k=="override_settings":
            x.setdefault("override_settings",{}); x["override_settings"].update(v)
        elif k in base or k in ALLOWED_KEYS:
            x[k]=v
    return x

def pick_keyword(kws:List[str])->str:
    return random.choice(kws) if kws else ""

def run_category(category:str):
    # 读取分类配置（脚本同目录 config_<category>.json）
    cfg_path = os.path.join(here(), f"config_{category}.json")
    cfg = load_json(cfg_path) or {}
    for k in list(cfg.keys()):
        if k not in ALLOWED_KEYS: cfg.pop(k,None)

    images_count = int(cfg.get("images_count", 20))
    base_prompt  = cfg.get("base_prompt","")
    kws = load_keywords(category)
    out_dir = build_output_dir(category)

    payload_base = merge_payload(DEFAULT_PAYLOAD, cfg)
    print(f"\n=== 分类 {category} | 目标 {images_count} 张 | 输出 {out_dir} ===")

    generated = 0
    file_idx = 1

    while generated < images_count:
        kw = pick_keyword(kws)
        prompt = (base_prompt + (", "+kw if kw else "")).strip() if base_prompt else kw
        payload = payload_base.copy()
        payload["prompt"] = prompt

        # 请求（带重试）
        for attempt in range(1, MAX_RETRY+1):
            try:
                r = requests.post(API_URL, json=payload, timeout=TIMEOUT)
                if r.status_code != 200:
                    print(f"[WARN] API {r.status_code}: {r.text[:200]}")
                    time.sleep(2*attempt); continue
                data = r.json()
                imgs = data.get("images",[])
                if not imgs:
                    print("[WARN] 无 images 字段，重试"); time.sleep(2*attempt); continue

                for img_b64 in imgs:
                    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{file_idx:02d}.jpg"
                    fpath = os.path.join(out_dir, fname)
                    b64_to_file(img_b64, fpath)
                    print(f"[OK] {category} -> {fname}   （{kw or '无关键词'}）")
                    generated += 1; file_idx += 1
                    if generated >= images_count: break
                break
            except Exception as e:
                print(f"[ERROR] 请求失败({attempt}/{MAX_RETRY})：{e}")
                time.sleep(2*attempt)
        else:
            print("[FATAL] 连续失败，跳过该分类")
            break

def main(cats:Optional[List[str]]=None):
    wait_api_ready()
    cats = cats or DEFAULT_CATEGORIES
    print("将按如下分类生成：", ", ".join(cats))
    for c in cats:
        run_category(c)
    print("\n全部完成。")

if __name__ == "__main__":
    main()
