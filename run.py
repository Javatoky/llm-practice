"""
练习文件运行入口
使用方法：python run.py Phase_A/system_prompt.py
"""
import sys
import runpy
from pathlib import Path

if len(sys.argv) < 2:
    print("用法：python run.py <练习文件路径>")
    print("例如：python run.py Phase_A/system_prompt.py")
    sys.exit(1)

target_file = sys.argv[1]
target_path = Path(target_file)

if not target_path.exists():
    print(f"错误：文件不存在 {target_file}")
    sys.exit(1)

module_path = str(target_path).replace("\\", ".").replace("/", ".").removesuffix(".py")
print(f"运行：{module_path}")
runpy.run_module(module_path, run_name="__main__")
