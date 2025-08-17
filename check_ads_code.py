import requests
from bs4 import BeautifulSoup

def check_ads_code(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        results = {
            "ads_js_found": False,
            "monetag_script_found": False,
            "zone_ids_found": [],
        }

        # 1. 是否引入了 ads.js
        scripts = soup.find_all("script")
        for script in scripts:
            src = script.get("src", "")
            if "ads.js" in src:
                results["ads_js_found"] = True
            if "fpyf8.com" in src:
                results["monetag_script_found"] = True
            if "data-zone" in str(script):
                results["zone_ids_found"].append(script.get("data-zone"))

        return results

    except Exception as e:
        return {"error": str(e)}


# ===== 修改为你的目标网址 =====
url_to_check = "http://g6.gogamefun.com"
result = check_ads_code(url_to_check)

print(f"\n🔍 检查网址：{url_to_check}\n")

if "error" in result:
    print(f"❌ 出错了：{result['error']}")
else:
    if result["ads_js_found"]:
        print("✅ 找到 ads.js 引用")
    else:
        print("❌ 没有找到 ads.js，请确认是否在 <head> 加入了 <script src='ads.js'>")

    if result["monetag_script_found"]:
        print("✅ 成功加载 Monetag 的广告脚本")
    else:
        print("❌ 没有找到 Monetag 脚本链接（可能被网络/VPN阻断）")

    if result["zone_ids_found"]:
        print(f"✅ 找到的 Zone ID：{', '.join(result['zone_ids_found'])}")
    else:
        print("❌ 没有发现任何 Monetag Zone ID 脚本段")

print("\n🧾 完成检测，请确认页面结构与加载网络环境。")
