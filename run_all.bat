@echo off
chcp 65001 >nul

echo Running: 2222.py (生成全部 HTML 页面 + SEO 字段占位) ...
python 2222.py

echo ----------------------------------------
echo Running: rebuild_index.py (重建 index.html 首页结构，来自模板 custom_homepage_template.html) ...
python rebuild_index.py

echo ----------------------------------------
echo Running: patch_homepage.py (插入 canonical、ads、schema 等 SEO 元素) ...
python patch_homepage.py

echo ----------------------------------------
echo Running: patch_struct_shuffle_and_lastmod.py (给所有页面插入结构 + lastmod 日期 + 打散顺序) ...
python patch_struct_shuffle_and_lastmod.py

echo ----------------------------------------
echo Running: generate_index.py (插入分类图片区块到主页 section 区域) ...
python generate_index.py

echo ----------------------------------------
echo Running: generate_link_list.py (生成全站图像链接索引页面 link_list.html) ...
python generate_link_list.py

echo ✅ 所有任务已完成!

