# -*- coding: utf-8 -*-
"""
不写死路径版：沿用“老脚本”的保存规则（脚本上一级目录 + 分类名），
并加入：分类配置、随机关键词、稳定出图参数、失败重试、统一命名/日志。
"""

import os
import json
import time
import random
import base64
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional

# ===== 基本配置 =====
API_URL   = "http://127.0.0.1:7860/sdapi/v1/txt2img"   # 需要 webui-user.bat 启动且 --api
TIMEOUT   = 180
MAX_RETRY = 3

# 默认要跑的分类（按需删/留）
DEFAULT_CATEGORIES = [
    "bedroom", "dark", "office", "soft", "uniform",
    "shower", "mirror", "fitness", "luxury", "redroom"
]

# ===== 工具函数 =====
def here() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def parent_of_here() -> str:
    return os.path.abspath(os.path.join(here(), ".."))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(path: str) -> Dict[str, Any]:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_base64_image(b64: str, out_file: str):
    data = base64.b64decode(b64)
    with open(out_file, "wb") as f:
        f.write(data)

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def filename_for(index: int) -> str:
    return f"{ts()}_{index:02d}.jpg"

def build_output_dir(category: str) -> str:
    """
    沿用老脚本的保存规则：脚本所在目录的上一级 + 分类名
    即：.../generator/../<category>/
    """
    out_dir = os.path.abspath(os.path.join(here(), "..", category))
    ensure_dir(out_dir)
    return out_dir

def merge_payload(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    x = defaults.copy()
    for k, v in overrides.items():
        if k == "override_settings":
            x.setdefault("override_settings", {})
            x["override_settings"].update(v)
        else:
            x[k] = v
    return x

# ====== 关键词池（可选） ======
def load_keywords() -> Dict[str, List[str]]:
    """
    读取 generator/keywords.json；没有就返回空。
    """
    kw_path = os.path.join(here(), "keywords.json")
    data = load_json(kw_path)
    result: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                result[k] = [s for s in (str(x).strip() for x in v) if s]
    return result

def pick_keyword(pool: List[str]) -> str:
    if not pool:
        return ""
    return random.choice(pool)

def build_prompt(base_prompt: str, kw: str) -> str:
    if base_prompt and kw:
        return f"{base_prompt}, {kw}"
    return base_prompt or kw

# ====== 默认的稳定出图参数（尽量减重影/融合）======
DEFAULT_PAYLOAD: Dict[str, Any] = {
    # prompt 由 base_prompt + 随机关键词 动态拼接
    "prompt": "",
    # 注意：这里负面词重点压“二人/多头/重影/融合”
    "negative_prompt": (
        "(worst quality, low quality, normal quality:1.2), "
        "text, watermark, logo, signature, blurry, lowres, "
        "multiple people, two persons, extra face, fused head, duplicated head, "
        "deformed anatomy, dislocated limbs, extra arms, extra legs, "
        "long neck, malformed hands, missing fingers, extra fingers, "
        "bad hands, bad feet, cropped, out of frame"
    ),
    "sampler_name": "DPM++ 2M Karras",
    "steps": 28,
    "cfg_scale": 6.5,
    "width": 832,      # 纵向竖图更稳
    "height": 1248,
    "seed": -1,
    "restore_faces": False,
    "save_images": False,          # 我们自己保存，避免多一份
    # 高分修复：先出底图，再放大细化，降低融合概率
    "enable_hr": True,
    "hr_scale": 1.6,
    "denoising_strength": 0.25,    # 可被分类配置覆盖
    "hr_upscaler": "R-ESRGAN 4x+ Anime6B",  # 本地有别名就改成对应名字
    # 兜底：不往 webui 默认输出目录里保存
    "override_settings": {
        "outdir_txt2img_samples": "",
        "outdir_txt2img_grids": "",
        "outdir_save": "",
        "CLIP_stop_at_last_layers": 2  # 相当于 clip_skip=2（对人物收敛更稳一些）
    },
    "batch_size": 1
}

# 允许在分类 JSON 里覆盖的字段
ALLOWED_KEYS = {
    "prompt", "negative_prompt", "sampler_name", "steps", "cfg_scale",
    "width", "height", "seed",
    "enable_hr", "hr_scale", "hr_upscaler", "denoising_strength",
    "batch_size", "override_settings"
}

def load_category_config(category: str) -> Dict[str, Any]:
    """
    读取 generator/config_<category>.json
    支持的字段参考 ALLOWED_KEYS；另外可加：
      - images_count: 生成张数（默认 20）
      - base_prompt: 该类的前缀提示词（与随机关键词拼接）
    """
    cfg_path = os.path.join(here(), f"config_{category}.json")
    raw = load_json(cfg_path)
    overrides = {k: raw[k] for k in raw.keys() if k in ALLOWED_KEYS}
    images_count = int(raw.get("images_count", 20))
    base_prompt  = str(raw.get("base_prompt", "")).strip()
    return {"overrides": overrides, "images_count": images_count, "base_prompt": base_prompt}

def txt2img(payload: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.post(API_URL, json=payload, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            else:
                print(f"[WARN] API {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[ERROR] request failed (attempt {attempt}/{MAX_RETRY}): {e}")
        time.sleep(2 * attempt)
    raise RuntimeError("API 重试失败，请确认 WebUI 已启动且带 --api")

def run_category(category: str, kw_map: Dict[str, List[str]]):
    out_dir  = build_output_dir(category)
    cfg      = load_category_config(category)
    pool     = kw_map.get(category, [])  # 该类的随机关键词池

    print(f"\n=== 分类 {category} | 目标 {cfg['images_count']} 张 | 输出 {out_dir} ===")

    images_done = 0
    index = 1

    while images_done < cfg["images_count"]:
        kw = pick_keyword(pool)
        prompt = build_prompt(cfg["base_prompt"], kw).strip()

        payload = merge_payload(DEFAULT_PAYLOAD, cfg["overrides"])
        payload["prompt"] = prompt

        try:
            resp = txt2img(payload)
            if "images" not in resp or not resp["images"]:
                print("[WARN] API 返回没有 images 字段，跳过")
                continue

            for img_b64 in resp["images"]:
                fname = filename_for(index)
                save_base64_image(img_b64, os.path.join(out_dir, fname))
                print(f"[OK] {category} -> {fname} （{kw or '无关键词'}）")
                images_done += 1
                index += 1
                if images_done >= cfg["images_count"]:
                    break

        except Exception as e:
            print(f"[ERROR] {category} 生成失败：{e}")
            # 失败继续循环（内部有重试）

def main(categories: Optional[List[str]] = None):
    print("[OK] WebUI API 就绪")
    cats = categories or DEFAULT_CATEGORIES
    print("将按如下分类生成：", ", ".join(cats))
    kw_map = load_keywords()  # 可选，没有也没关系
    for c in cats:
        run_category(c, kw_map)
    print("\n全部完成。")

if __name__ == "__main__":
    # 你也可以只跑几个： main(["bedroom","soft"])
    main()
