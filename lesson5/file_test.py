import os

# 1. 打印当前脚本到底在哪运行
print(f"当前工作目录 (CWD): {os.getcwd()}")

# 2. 定义你的目标路径
target_path = "/home/neko/project/pyProject/langChainLearning/lesson5/test_manual.txt"

# 3. 尝试写入
try:
    with open(target_path, "w") as f:
        f.write("这是手动测试，证明WSL没有坏。")
    print(f"✅ 成功写入: {target_path}")
except Exception as e:
    print(f"❌ 写入失败! 原因: {e}")