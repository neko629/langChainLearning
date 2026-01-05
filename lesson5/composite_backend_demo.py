import asyncio
import shutil
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from model_factory import get_model

# DeepAgents 导入
from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, ToolMessage


# 导入 DockerBackend
try:
    from docker_backend_demo import DockerBackend
except ImportError:
    try:
        from deepagents.backends.docker import DockerBackend
    except ImportError:
        DockerBackend = None

def print_header():
    print("\n" + "="*80)
    print("DeepAgents CompositeBackend 混合后端演示 (极简版)")
    print("架构：混合云原生模式 (Docker 执行 + 本地持久化)")
    print("="*80)

async def setup_mcp_tools():
    print("  → 正在连接 Context7 MCP 服务器...")
    try:
        client = MultiServerMCPClient({
            "context7": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp@latest"],
            }
        })
        tools = await client.get_tools()
        print("  → MCP 工具加载成功")
        return client, tools
    except Exception as e:
        print(f"ERROR: MCP 连接失败: {e}")
        return None, []

async def run_composite_demo():
    load_dotenv(override=True)
    print_header()

    if DockerBackend is None:
        print("严重错误: 未找到 DockerBackend。请确保 docker_backend.py 存在。")
        return

    # Step 1
    print("\n" + "-"*40)
    print("步骤 1: 配置混合环境")
    print("-"*40)

    host_work_dir = Path("workspace/data_analysis_project").resolve()
    if host_work_dir.exists():
        shutil.rmtree(host_work_dir)
    host_work_dir.mkdir(parents=True, exist_ok=True)
    print(f"  • 宿主机持久层: {host_work_dir}")

    container_mount_path = "/data"
    docker_volumes = {
        str(host_work_dir): {'bind': container_mount_path, 'mode': 'rw'}
    }
    print(f"  • 容器挂载:      {host_work_dir} ↔ {container_mount_path}")

    # Step 2
    print("\n" + "-"*40)
    print("步骤 2: 初始化混合后端 (Composite Backend)")
    print("-"*40)

    fs_backend = FilesystemBackend(root_dir=host_work_dir, virtual_mode=True)

    print("  • 正在启动 Docker 容器 (python:3.11-slim)...")
    docker_backend = DockerBackend(
        image="python:3.11-slim",
        auto_remove=True,
        volumes=docker_volumes
    )

    routes = {
        container_mount_path: fs_backend
    }
    backend = CompositeBackend(default=docker_backend, routes=routes)

    print("\n[路由表配置]")
    print(f"1. 默认路由 (/): DockerBackend (临时执行)")
    print(f"2. 持久化路由 ({container_mount_path}/*): FilesystemBackend (宿主机存储)")

    # Step 3
    print("\n" + "-"*40)
    print("步骤 3: 部署 Agent")
    print("-"*40)

    mcp_client, mcp_tools = await setup_mcp_tools()

    system_prompt = f"""你是一名在混合环境中工作的高级数据工程师。

    环境地图:
    1. 执行层 (根目录 `/`):
       - 临时的 Docker 容器。
       - 用于创建脚本 (`.py`) 和运行命令。
       - 这里的文会在会话结束后消失。

    2. 存储层 (`{container_mount_path}`):
       - 从宿主机挂载的持久化存储。
       - 用于存放 输入 数据和 输出 报告。
       - 这里的文件会永久保存。

    你的任务:
    1. **摄入**: 创建一个文件 `{container_mount_path}/raw_metrics.txt`，内容为 "CPU: 45%, Mem: 60%"。
       (注意: 这使用了 'write_file' 工具，该工具通过路由直接写入宿主机文件系统)。

    2. **处理**: 创建一个 Python 脚本 `/processor.py` (在根目录)，该脚本:
       - 读取 `{container_mount_path}/raw_metrics.txt`。
       - 计算 "健康分数" (模拟一下即可)。
       - 将报告写入 `{container_mount_path}/health_report.txt`。
       - 打印 "Analysis Complete"。

    3. **执行**: 使用 `python /processor.py` 运行脚本。
       (注意: 这在 Docker 内部运行。Docker 因为卷挂载能看到这些文件)。

    4. **验证**: 读取 `{container_mount_path}/health_report.txt` 并显示它。
    """

    agent = create_deep_agent(
        model=get_model(),
        tools=mcp_tools,
        backend=backend,
        system_prompt=system_prompt
    )

    # Step 4
    print("\n" + "-"*40)
    print("步骤 4: 任务执行")
    print("-"*40)

    task_input = "开始工程流水线。"
    config = {"configurable": {"thread_id": "composite_demo_simple_v1"}}

    step_count = 0
    try:
        message_history_len = 0

        async for event in agent.astream({"messages": [("user", task_input)]}, config=config):
            if "messages" in event:
                current_messages = event["messages"]
                if len(current_messages) > message_history_len:
                    for i in range(message_history_len, len(current_messages)):
                        msg = current_messages[i]

                        # Agent Thinking
                        if isinstance(msg, BaseMessage) and msg.content and not getattr(msg, "tool_calls", None):
                            step_count += 1
                            print(f"\n[🧠 Agent 思考 (步骤 {step_count})]:\n{msg.content}")

                        # Tool Calls
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            step_count += 1
                            for tc in msg.tool_calls:
                                tool_name = tc['name']
                                args = tc['args']

                                # Routing logic visualization
                                target = "Docker 容器 🐳"
                                path_arg = args.get('file_path') or args.get('path')
                                if path_arg and str(path_arg).startswith(container_mount_path):
                                    target = "宿主机文件系统 💾"

                                print(f"\n[🛠️ 工具执行 (步骤 {step_count})]:")
                                print(f"  • 工具: {tool_name}")

                                # Special handling for code content
                                if tool_name == "write_file" and path_arg and str(path_arg).endswith(".py"):
                                    code_content = args.get("content", "")
                                    # Print args without content first
                                    args_copy = args.copy()
                                    args_copy['content'] = "(代码内容如下...)"
                                    print(f"  • 参数: {args_copy}")
                                    print(f"  • 路由: → {target}")
                                    print(f"  • 📝 写入代码内容:\n")
                                    print("-" * 20)
                                    print(code_content)
                                    print("-" * 20)
                                else:
                                    print(f"  • 参数: {str(args)[:200] + '...' if len(str(args)) > 200 else args}")
                                    print(f"  • 路由: → {target}")

                        # Tool Outputs
                        if isinstance(msg, ToolMessage):
                            content = msg.content
                            if len(content) > 300:
                                content = content[:300] + "... [已截断]"
                            print(f"\n[↳ 输出]: {content}")

                    message_history_len = len(current_messages)

    except Exception as e:
        print(f"\n运行时错误: {e}")

    # Step 5
    print("\n" + "-"*40)
    print("步骤 5: 宿主机侧验证")
    print("-"*40)

    report_path = host_work_dir / "health_report.txt"
    raw_path = host_work_dir / "raw_metrics.txt"

    if raw_path.exists():
        print(f"✅ 原始数据已找到: {raw_path} (通过直接 FS 路由创建)")
    else:
        print(f"❌ 原始数据丢失: {raw_path}")

    if report_path.exists():
        content = report_path.read_text()
        print(f"\n🏆 持久化验证成功! 文件: {report_path}")
        print("内容:")
        print("-" * 20)
        print(content)
        print("-" * 20)
    else:
        print(f"❌ 报告丢失: {report_path}")

    # Step 6
    print("\n正在关闭基础设施...")
    if 'docker_backend' in locals() and hasattr(docker_backend, "close"):
        docker_backend.close()
        print("  • Docker 容器已终止")

    print("\n✨ 演示圆满完成！")

if __name__ == "__main__":
    try:
        asyncio.run(run_composite_demo())
        #await run_composite_demo()
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print("检测到正在运行的事件循环。请在单元格中使用 'await run_composite_demo()'。")
        else:
            raise e
