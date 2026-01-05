import asyncio
import os
import shutil
from pathlib import Path
from langchain_core.messages import BaseMessage, ToolMessage
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from model_factory import get_model


# 定义工作目录
WORK_DIR = Path("agent_data").resolve()
LARGE_FILE_NAME = "server_logs.txt"
LARGE_FILE_PATH = WORK_DIR / LARGE_FILE_NAME
TARGET_SECRET = "CRITICAL_ERROR_CODE_998877"

async def run_pagination_demo():
    print("\n" + "="*80)
    print("DeepAgents FilesystemMiddleware 分页读取演示")
    print("="*80)

    # 2. 初始化 FilesystemBackend
    # 将 backend 指向我们的测试目录
    # virtual_mode=True 确保 Agent 只能访问该目录下的文件，不能访问宿主机其他目录
    backend = FilesystemBackend(root_dir=WORK_DIR, virtual_mode=True)

    # 3. 创建 Agent
    # 我们明确指示 Agent 使用分页读取，每次读取 300 行，Agent默认每次读取500行(DEFAULT_READ_LIMIT)
    system_prompt = """
    你是一个专业的系统管理员。
    你的任务是从日志文件中查找特定的错误代码。
    注意：日志文件可能非常大，为了避免上下文溢出，你必须使用 `read_file` 工具的分页功能。
    每次读取请限制在 300 行以内 (limit=300)，并使用 offset 参数向后滚动。
    直到找到目标信息为止。
    """

    llm = get_model('deepseek-chat', 'deepseek', temperature=0, timeout=300)

    agent = create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt=system_prompt
    )

    task = f"请在 '/{LARGE_FILE_NAME}' 中查找包含 '{TARGET_SECRET}' 的行，并告诉我它的具体内容。"

    print(f"\n任务: {task}")
    print("-" * 60)

    # 4. 执行任务并观察分页行为
    step = 0
    print("\n开始流式输出...")
    try:
        async for event in agent.astream({"messages": [("user", task)]}):
            for node_name, node_data in event.items():
                # debug: print(f"DEBUG: Node: {node_name}")
                if not node_data: continue

                # 处理 Overwrite 对象
                if hasattr(node_data, "value"):
                    node_data = node_data.value

                if not isinstance(node_data, dict):
                    continue

                if "messages" in node_data:
                    msgs = node_data["messages"]
                    if hasattr(msgs, "value"):
                        msgs = msgs.value

                    if not isinstance(msgs, list): msgs = [msgs]

                    for msg in msgs:
                        # 1. 打印 Agent 的思考 (AIMessage with tool_calls)
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            step += 1
                            print(f"\n[Step {step}] Agent 决定调用工具 (Node: {node_name}):")
                            for tc in msg.tool_calls:
                                name = tc['name']
                                args = tc['args']
                                print(f"  >>> 工具: {name}")

                                if name == "read_file":
                                    offset = args.get('offset', 0)
                                    limit = args.get('limit', 'Default')
                                    path_val = args.get('path') or args.get('file_path')
                                    print(f"  >>> 参数: path='{path_val}', offset={offset}, limit={limit}")
                                    print(f"      (说明: 正在读取从第 {offset} 行开始的 {limit} 行数据)")
                                else:
                                    print(f"  >>> 参数: {args}")

                        # 2. 打印工具的输出 (ToolMessage)
                        elif isinstance(msg, ToolMessage):
                            content = msg.content
                            line_count = len(content.splitlines())
                            # 检查是否包含目标 Secret
                            found_secret = TARGET_SECRET in content

                            preview = content[:100].replace('\n', ' ') + "..."
                            print(f"\n[Tool Output] (Node: {node_name}) 读取了 {line_count} 行数据")
                            print(f"  内容预览: {preview}")
                            if found_secret:
                                print(f"  ✨ 成功: 在此分块中发现了目标 Secret: {TARGET_SECRET}")

                        # 3. 打印 Agent 的最终回复 (AIMessage without tool_calls)
                        elif isinstance(msg, BaseMessage) and msg.type == "ai" and msg.content:
                            print(f"\n[Agent 最终回复] (Node: {node_name}):")
                            print("-" * 40)
                            print(msg.content)
                            print("-" * 40)


    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_pagination_demo())
