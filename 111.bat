@echo off
chcp 65001 >nul
echo ✅ 当前终端编码设置为 UTF-8（防止中文乱码）

REM === 进入 generator 文件夹，执行图片生成 ===
cd /d "%~dp0generator"
echo 🖼️ 正在批量生成图片...
call run_generator_autopath.bat

REM === 返回主目录，执行网页生成+SEO结构 ===
cd ..
echo 🌐 正在执行网页生成 + SEO 插入...
call run_all.bat

REM === 上传到 GitHub（你已经配置了 SSH 公钥） ===
echo 🚀 正在上传到 GitHub 仓库...
python auto_git_push.py

echo ✅ 全部流程执行完毕！
pause
