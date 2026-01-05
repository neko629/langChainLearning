import asyncio
import os
import sys
import uuid

from deepagents.backends import StoreBackend
from dotenv import load_dotenv
from asyncio.subprocess import PIPE
import shutil

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic.v1 import BaseModel

from model_factory import get_model

from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from deepagents import create_deep_agent

load_dotenv(override=True)


from langchain_mcp_adapters.client import MultiServerMCPClient

DB_URI = "postgresql://postgres:123456@localhost:5432/store_backend"

def print_header():
    print("\n" + "="*80)
    print("DeepAgents StoreBackend (PostgreSQL) 演示 (极简版)")
    print("="*80)


async def setup_mcp_tools():
    """
    连接 Context7 MCP 服务器并获取工具。
    """
    print("Step 1.1: 正在连接 Context7 MCP 服务器...")
    try:
        # 使用官方 Context7 MCP 配置
        client = MultiServerMCPClient({
            "context7": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp@latest"],
            }
        })

        # 获取工具列表
        tools = await client.get_tools()

        print(f"Step 1.2: 成功加载 {len(tools)} 个 MCP 工具")
        return client, tools
    except Exception as e:
        print(f"ERROR: 连接 MCP 失败: {e}")
        return None, []

async def run_store_backend_demo():
    print_header()

    mcp_client, tools = await setup_mcp_tools()

    print(f"Step 2: 初始化 StoreBackend 连接到 PostgreSQL 数据库...")

    try:
        with ConnectionPool(conninfo = DB_URI, kwargs = {"autocommit": True}) as pool:

            checkpointer = MemorySaver()
            store = PostgresStore(pool)

            with pool.connection() as conn:
                with conn.cursor() as cur:
                    for migration in store.MIGRATIONS:
                        cur.execute(migration)
            print("Step 2: StoreBackend 初始化成功。")

            model = get_model()

            backend_factory = lambda rt: StoreBackend(rt)
            print("Step 3: 创建 Deep Agent...")

            agent = create_deep_agent(
                model = model,
                tools = tools,
                backend = backend_factory,
                store = store,
                checkpointer = checkpointer,
                system_prompt="""你是一个高级技术助手。
                            你的任务是使用 Context7 工具查询关于 'DeepAgents StoreBackend' 的用法。
                            查询后，创建一个总结文件 '/knowledge/store_backend_notes.md'，并写入关键信息。
                            由于你使用的是 StoreBackend，这个文件将直接存储在 PostgreSQL 数据库中。
                            最后，请读取该文件以验证存储成功。"""
            )

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            task = "请查询 StoreBackend 的用法，并将总结写入 /knowledge/store_backend_notes.md，最后读取它验证。"

            print(f"Session Thread ID: {thread_id}")
            print("\n" + "-" * 40)
            print(f"Step 4: 开始执行任务: {task}")
            print("-" * 40)

            step = 0
            message_history_len = 0

            try:
                async for event in agent.astream(
                    {
                        "messages": [
                            ("user", task)
                        ]
                    },
                    config = config
                ):
                    if "messages" in event:
                        current_messages = event["messages"]
                        if len(current_messages) > message_history_len: # 新消息
                            for i in range(message_history_len, len(current_messages)):
                                msg = current_messages[i]

                                # 普通文本消息
                                if isinstance(msg, BaseMessage) and msg.content:
                                    is_tool_call = getattr(msg, "tool_calls", None)
                                    if not is_tool_call:
                                        print(f"\n[Step {step}] Agent 思考:\n{msg.content}\n")
                                # 工具调用
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    step += 1
                                    print(f"\n[Step {step}] Agent 决定调用工具:")
                                    for tc in msg.tool_calls:
                                        tool_name = tc['name']
                                        tool_args = tc['args']
                                        print(f"  >>> 工具: {tool_name}")
                                        print(f"  >>> 参数: {tool_args}")

                                # 工具消息
                                if isinstance(msg, ToolMessage):
                                    content_preview = msg.content[:200] + ... if len(msg.content) > 200 else msg.content
                                    print(f"\n[Step {step}] 工具返回:\n{content_preview}\n")
                            message_history_len = len(current_messages)
            except Exception as e:
                print(f"[错误] 运行 Agent 时出错: {e}")
                import traceback
                traceback.print_exc()

            print("\n" + "-" * 40)
            print("Setp 5: 验证 StoreBackend 持久化")
            print("\n" + "-" * 40)

            print("\n正在从数据库中读取 '/knowledge/store_backend_notes.md' 文件内容...")

            read_agent = create_deep_agent(
                model = model,
                backend = backend_factory,
                system_prompt = """你是一个文件读取助手。""",
                checkpointer = checkpointer,
                store = store
            )
            read_task = "请读取 '/knowledge/store_backend_notes.md' 文件的内容，并展示给我。"

            read_result = await read_agent.ainvoke(
                {
                    "messages": [
                        ("user", read_task)
                    ]
                },
                config = config
            )
            last_msg = read_result["messages"][-1]

            print("\n文件内容如下:\n")
            print(last_msg.content)
            print("=" * 40)
            print("使用sql命令直接验证文件存储...")
            sql_query = """
                SELECT
                    key, value ->>'content' as content, updated_at
                FROM store
                WHERE prefix = 'filesystem'
                  AND key = '/knowledge/store_backend_notes.md'; \
                """
            print(sql_query.strip()) # 打印SQL语句
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query)
                    row = cur.fetchone()
                    if row:
                        key, content, updated_at = row
                        print(f"\n查询结果:\n文件路径: {key}\n最后更新: {updated_at}\n内容预览:\n{content[:500]}{'...' if len(content) > 500 else ''}")
                    else:
                        print("未找到对应的文件记录。")
            print("演示结束")
    except Exception as e:
        print(f"[错误] 初始化 StoreBackend 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_store_backend_demo())




