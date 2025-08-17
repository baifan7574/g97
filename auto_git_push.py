import subprocess
import time

MAX_RETRIES = 3
DELAY_SECONDS = 10

def run_git_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print(result.stdout.decode())
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 错误信息：", e.stderr.decode())
        return False

def push_with_retry():
    print("📦 开始执行 Git 自动上传流程...\n")
    
    subprocess.run("git add .", shell=True)
    subprocess.run('git commit -m "Auto update"', shell=True)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"🚀 第 {attempt} 次尝试 git push...")
        success = run_git_command("git push")
        if success:
            print("✅ 上传成功！")
            break
        else:
            if attempt < MAX_RETRIES:
                print(f"🔁 上传失败，{DELAY_SECONDS} 秒后重试...\n")
                time.sleep(DELAY_SECONDS)
                DELAY_SECONDS += 10  # 每次多等10秒
            else:
                print("❌ 所有尝试都失败了，请检查网络或稍后再试。")

if __name__ == "__main__":
    push_with_retry()
