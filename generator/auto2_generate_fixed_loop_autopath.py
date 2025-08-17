
import os
import json
import requests
from datetime import datetime

def generate_images(config_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_file)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    prompt = config["prompt"]
    negative_prompt = config["negative_prompt"]
    steps = config["steps"]
    sampler_index = config["sampler_index"]
    width = config["width"]
    height = config["height"]
    batch_size = config["batch_size"]
    category = config["category"]

    output_dir = os.path.abspath(os.path.join(current_dir, "..", category))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(batch_size):
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "sampler_index": sampler_index,
            "width": width,
            "height": height,
        }

        try:
            response = requests.post(url="http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()
            r = response.json()
            image_data = r['images'][0]

            import base64
            image_bytes = base64.b64decode(image_data.split(",", 1)[0] if "," in image_data else image_data)
            file_path = os.path.join(output_dir, f"{timestamp}_{i+1:02d}.jpg")
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            print(f"✅ 已保存：{file_path}")
        except Exception as e:
            print("❌ 生成失败：", e)

if __name__ == "__main__":
    for file in os.listdir():
        if file.startswith("config_") and file.endswith(".json"):
            print(f"\n📦 正在处理配置：{file}")
            generate_images(file)
