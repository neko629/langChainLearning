import asyncio
import os
import sys
from deepagents import create_deep_agent
from langchain_core.messages import BaseMessage

from model_factory import get_model

# 尝试导入 DockerBackend
try:
    from docker_backend_demo import DockerBackend
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from docker_backend_demo import DockerBackend

async def run_docker_demo():
    print("\n" + "=" * 80)
    print("DeepAgents DockerBackend 演示 (极简版)")
    print("=" * 80)

    try:
        import docker
        docker.from_env().ping()
    except Exception as e:
        print(f"[错误] 无法连接到 Docker 守护进程: {e}")
        print("请确保 Docker 已安装并正在运行。")
        return
    # 初始化 DockerBackend
    try:
        backend = DockerBackend(
            image="python:3.11-slim",
            auto_remove=True
        )
        print(f"[成功] 已连接到 Docker 容器: {backend.id}")
    except Exception as e:
        print(f"[错误] 无法初始化 DockerBackend: {e}")
        return

    # 创建 Agent
    try:
        llm = get_model('deepseek-chat', 'deepseek', temperature=0.7, timeout=300)
        system_prompt = """
        你是一个运行在 Docker 容器中的 AI 助手。
            你的任务是演示环境隔离性。

            请执行以下步骤：
            1. 运行 'cat /etc/os-release' 查看容器操作系统。
            2. 运行 'python --version' 确认 Python 环境。
            3. 创建文件 '/workspace/hello_docker.py'，内容为打印 'Hello from Docker Container!'。
            4. 运行该脚本。
        """
        agent = create_deep_agent(
            model=llm,
            backend=backend,
            system_prompt=system_prompt
        )
        print("[成功] Agent 创建完成。")

        async for event in agent.astream(
                {
                    "messages": [
                        (
                            "user",
                            "请按照系统提示执行任务，展示 Docker 容器内的操作环境隔离性。"
                        )
                    ]
                }
        ):
            for node_name, node_data in event.items():
                if not node_data:
                    continue
                if hasattr(node_data, "value"):
                    node_data = node_data.value

                if not isinstance(node_data, dict):
                    continue

                if "messages" in node_data:
                    messages = node_data["messages"]
                    if hasattr(messages, "value"):
                        messages = messages.value
                    if isinstance(messages, list) and messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, BaseMessage) and last_msg.content:

                            print(f"\n[Node: {node_name}] Agent 回复:\n{last_msg.content}")
    except Exception as e:
        print(f"[错误] 运行 Agent 时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n正在清理容器...")
        backend.close()
        print("容器已移除")
        print("\n演示结束")

if __name__ == "__main__":
    asyncio.run(run_docker_demo())