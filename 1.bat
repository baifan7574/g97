@echo off
:: 设置控制台编码为 UTF-8，防止中文乱码
chcp 65001 > nul

:: 激活 Python 虚拟环境（如果有）或直接使用系统 Python
:: 如果你使用了虚拟环境，把下面这一行取消注释并修改路径
:: call venv\Scripts\activate

:: 运行 Python 脚本
python 1.py

:: 保持窗口不关闭，方便查看结果
pause
