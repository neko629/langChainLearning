# %%
import os
from dotenv import load_dotenv
import time
import uuid
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent, AgentState
from typing import Annotated
from pydantic import BaseModel, Field
from model_factory import get_model

# --- 核心组件：Postgres 持久化检查点 ---
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore, InjectedState
from psycopg_pool import ConnectionPool

# from langchain_community.storage import MongoDBStore

# 配置 API KEY
load_dotenv(override=True)

# ==========================================
# 1. 数据库配置
# ==========================================
# 对应上面 Docker 启动命令的配置
DB_URL = "postgresql://postgres:123456@localhost:5432/langchain_db"
llm = get_model("qwen2.5:7b", "ollama")


# ==========================================
# 2. 定义工具 (Tools)
# ==========================================
@tool
def magic_calculation(a: int, b: int) -> int:
    """进行一次特殊的加法计算"""
    return (a + b) * 10


"""
跨线程记忆需要传递 user_id，通过自定义 State 实现
"""


class CrossThreadState(AgentState):
    user_id: str  # 跨线程记忆的唯一标识


# ============ 定义 Pydantic 模型用于提取用户信息 ===========
class UserInfo(BaseModel):
    """从文本中提取的用户信息"""
    user_name: str = Field(description="用户的名字，例如：Alice、Bob、张三等")
    additional_info: str = Field(description="关于用户的其他信息，例如职业、兴趣爱好等")


class QueryInfo(BaseModel):
    """从查询文本中提取的信息"""
    user_name: str = Field(
        description="要查询的用户名字。如果查询中包含'我的'、'我是'等第一人称，请从对话历史中提取用户名；如果没有明确的用户名，返回'all_users'"
    )
    query_content: str = Field(description="查询的具体内容，例如：职业、兴趣爱好等")


# ============ 定义记忆管理工具（使用 BaseStore）===========
# 注意：store 参数使用 InjectedStore 注解，由 LangGraph 自动注入
# InjectedStore() 标记会让 Pydantic 在生成 JSON Schema 时跳过这个参数
# LLM 不会看到 store 参数，只会看到 user_id 和 info
@tool
def remember_user_info(
        info: str,
        state: Annotated[dict, InjectedState()],
        store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """
    将用户信息存入跨线程记忆

    重要：此工具会自动从 state 中获取 user_id，并使用 Pydantic 提取用户信息

    参数说明：
        info: 要记忆的信息（例如：用户的名字、职业、偏好等）

    示例：
        - remember_user_info("用户名叫 Alice，是一名工程师")
        - remember_user_info("我是 Bob，喜欢深度学习")
    """
    # 使用 Pydantic 提取用户信息
    structured_llm = llm.with_structured_output(UserInfo) # 创建结构化输出 LLM

    try:
        # 从文本中提取结构化信息
        extracted_info = structured_llm.invoke(
            f"从以下文本中提取用户名和其他信息：{info}"
        )

        # 优先使用提取的用户名，如果提取失败则使用 state 中的 user_id
        extracted_user_name = extracted_info.user_name.lower()
        state_user_id = state.get("user_id", "unknown_user")

        # 如果提取到的用户名不是 unknown，则使用提取的用户名
        if extracted_user_name and extracted_user_name != "unknown":
            user_id = extracted_user_name
        else:
            user_id = state_user_id

        full_info = f"{extracted_info.user_name}: {extracted_info.additional_info}"

    except Exception as e:
        # 如果提取失败，使用 state 中的 user_id
        user_id = state.get("user_id", "unknown_user")
        full_info = info

    # 命名空间设计：(用户ID, 信息类别)
    namespace = (user_id, "profile")

    # 生成唯一记忆ID
    memory_id = str(uuid.uuid4())

    # 存储到 BaseStore（自动持久化）
    store.put(
        namespace,
        memory_id,
        {
            "info": full_info,
            "timestamp": "2025-11-25",
            "source": "user_input"
        }
    )

    return f"✅ 已将信息存入长期记忆 (用户: {user_id}): {full_info}"


@tool
def recall_user_info(
        query: str,
        state: Annotated[dict, InjectedState()],
        store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """
    从跨线程记忆中检索用户信息

    重要：此工具会自动从 state 中获取 user_id

    参数说明：
        query: 查询关键词（用于描述要查找的信息，例如"我的职业"、"我的兴趣"等）

    返回：用户的历史信息
    """
    # 优先从 state 中获取 user_id
    state_user_id = state.get("user_id", None)

    # 如果 state 中没有 user_id，尝试使用 Pydantic 从查询中提取
    if not state_user_id:
        structured_llm = llm.with_structured_output(QueryInfo)

        try:
            # 从查询文本中提取结构化信息
            prompt = f"""从以下查询中提取用户名和查询内容。

                    查询文本：{query}

                    注意：
                    1. 如果查询中包含"我的"、"我是"等第一人称词汇，说明用户在询问自己的信息
                    2. 如果能从查询中推断出具体的用户名（如 Alice、Bob），请提取该用户名
                    3. 如果无法确定具体用户，返回 'all_users'
            """
            extracted_query = structured_llm.invoke(prompt)

            # 如果提取到的是 'all_users'，则搜索所有用户
            if extracted_query.user_name.lower() in ['all_users', 'current_user',
                                                     'unknown']:
                user_id = None
            else:
                user_id = extracted_query.user_name.lower()

        except Exception as e:
            # 如果提取失败，搜索所有用户
            user_id = None
    else:
        # 使用 state 中的 user_id
        user_id = state_user_id

    try:
        if user_id:
            # 使用 namespace_prefix 搜索该用户的所有记忆
            namespace_prefix = (user_id,)
            memories = store.search(namespace_prefix, limit=20)
        else:
            # 搜索所有已知用户的记忆
            memories = []
            # 先尝试获取所有可能的用户
            for uid in ['alice', 'bob', 'unknown_user']:
                namespace_prefix = (uid,)
                user_memories = store.search(namespace_prefix, limit=20)
                memories.extend(user_memories)

        if not memories:
            return f"未找到相关记忆。请先告诉我一些信息，我会记住它们。"

        # 格式化返回
        results = []
        for item in memories:
            info = item.value.get('info', '未知信息')
            timestamp = item.value.get('timestamp', '未知时间')
            results.append(f"- {info} (记录时间: {timestamp})")

        return f"找到 {len(results)} 条记忆:\n" + "\n".join(results)

    except Exception as e:
        return f"检索记忆时出错: {str(e)}"


# %%
# ==========================================
# 3. 主程序逻辑
# ==========================================
def run_postgres_agent():
    print("--- 正在连接 PostgreSQL 数据库 ---")

    # 使用 ConnectionPool 管理数据库连接
    # PostgresSaver 需要在这个上下文管理器中运行
    with ConnectionPool(conninfo=DB_URL, max_size=20,
                        kwargs={"autocommit": True}) as pool:

        # --- A. 初始化 Checkpointer 和 Store ---
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool)

        # 注意：第一次运行时需要创建表结构，会检测数据库，如果不存在，会自动创建所需的表结构。
        # 生产环境只需运行一次，但在脚本中加上是安全的（幂等操作）。
        # checkpointer 会创建 'checkpoints', 'checkpoint_blobs' 等表
        # store 会创建 'store' 表
        print("🔧 初始化 Checkpointer 表结构...")
        checkpointer.setup()
        print("✅ Checkpointer 表结构初始化完成")

        print("🔧 初始化 Store 表结构...")
        store.setup()
        print("✅ Store 表结构初始化完成")

        # --- B. 创建 Agent ---

        # 包含所有工具：计算工具 + 跨线程记忆工具
        tools = [magic_calculation, remember_user_info, recall_user_info]

        agent = create_agent(
            model=llm,
            tools=tools,
            state_schema=CrossThreadState,  # 自定义状态传递 user_id
            system_prompt="""
            你是一个具备跨线程记忆的智能助手。

            你的能力：
            1. 使用 remember_user_info 工具将用户的重要信息存入长期记忆（跨会话持久化）
            2. 使用 recall_user_info 工具从长期记忆中检索用户信息
            3. 使用 magic_calculation 工具进行特殊计算

            工作流程：
            - 当用户告诉你他的名字、职业、偏好等信息时，主动调用 remember_user_info 存储
            - 当用户询问"你还记得我吗"或类似问题时，调用 recall_user_info 检索
            - 记忆是跨会话的，即使在新的对话中也能记住用户信息

            注意：调用 remember_user_info 和 recall_user_info 时，必须传入 user_id 参数（从 state 中获取）。
            """,
            store=store,  # ✅ 注入 BaseStore 实现跨线程记忆
            checkpointer=checkpointer  # 注入数据库检查点（单会话记忆）
        )

        # ==========================================
        # 4. 测试场景：跨线程记忆功能
        # ==========================================

        print("\n" + "=" * 70)
        print("场景 1：用户 Alice 第一次对话（会话 1）")
        print("=" * 70)

        # 1. Alice 的第一个会话
        thread1_config = {"configurable": {"thread_id": "session_alice_001"}}

        print("\n👤 用户 Alice: 你好，我是 Alice，一名 Python 开发工程师，我喜欢深度学习。")

        for chunk in agent.stream(
                {
                    "messages": [HumanMessage(
                        content="你好，我是 Alice，一名 Python 开发工程师，我喜欢深度学习。")],
                    "user_id": "alice"  # 传入 user_id
                },
                config=thread1_config,
                stream_mode="values"
        ):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"   🔧 [调用工具]: {tool_call['name']}")

        print("\n" + "=" * 70)
        print("场景 2：用户 Alice 第二次对话（会话 2 - 不同 thread_id）")
        print("=" * 70)
        print("💡 模拟：Alice 关闭浏览器，第二天重新打开，开始新会话")
        time.sleep(1)

        # 2. Alice 的第二个会话（不同的 thread_id）
        thread2_config = {"configurable": {"thread_id": "session_alice_002"}}

        print("\n👤 用户 Alice: 你还记得我是谁吗？我的职业是什么？")

        for chunk in agent.stream(
                {
                    "messages": [
                        HumanMessage(content="你还记得我是谁吗？我的职业是什么？")],
                    "user_id": "alice"  # ✅ 同样的 user_id
                },
                config=thread2_config,
                stream_mode="values"
        ):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"   🔧 [调用工具]: {tool_call['name']}")

        print("\n" + "=" * 70)
        print("场景 3：用户 Bob 的对话（不同用户）")
        print("=" * 70)

        # 3. Bob 的会话
        thread3_config = {"configurable": {"thread_id": "session_bob_001"}}

        print("\n👤 用户 Bob: 你好，我是 Bob，一名产品经理，帮我算一下 10 + 20 的特殊结果。")

        for chunk in agent.stream(
                {
                    "messages": [HumanMessage(
                        content="你好，我是 Bob，一名产品经理，帮我算一下 10 + 20 的特殊结果。")],
                    "user_id": "bob"  # ✅ 不同的 user_id
                },
                config=thread3_config,
                stream_mode="values"
        ):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"   🔧 [调用工具]: {tool_call['name']}")

        print("\n" + "=" * 70)
        print("场景 4：Alice 第三次对话（验证记忆隔离）")
        print("=" * 70)
        print("💡 验证：Alice 的记忆不会被 Bob 的信息污染")

        # 4. Alice 的第三个会话
        thread4_config = {"configurable": {"thread_id": "session_alice_003"}}

        print("\n👤 用户 Alice: 我的兴趣爱好是什么？")

        for chunk in agent.stream(
                {
                    "messages": [HumanMessage(content="我的兴趣爱好是什么？")],
                    "user_id": "alice"
                },
                config=thread4_config,
                stream_mode="values"
        ):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"   🔧 [调用工具]: {tool_call['name']}")

        print("\n" + "=" * 70)
        print("✅ 跨线程记忆测试完成！")
        print("=" * 70)
        print("\n 测试总结:")
        print("  ✅ 场景 1: Alice 首次对话，Agent 自动存储用户信息到 Store")
        print("  ✅ 场景 2: Alice 新会话（不同 thread_id），Agent 成功从 Store 检索记忆")
        print("  ✅ 场景 3: Bob 的对话，Agent 为 Bob 创建独立的记忆空间")
        print("  ✅ 场景 4: Alice 再次对话，记忆未被 Bob 的信息污染")
        print("\n💡 关键特性:")
        print("  - Checkpointer: 管理单个会话的对话历史（基于 thread_id）")
        print("  - Store: 管理跨会话的长期记忆（基于 user_id）")
        print("  - 记忆隔离: 不同用户的记忆完全隔离（通过 namespace）")
        print("  - 持久化: 所有数据存储在 PostgreSQL，重启程序后依然可用")
        print("=" * 70)

run_postgres_agent()
