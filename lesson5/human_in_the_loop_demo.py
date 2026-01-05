import os
import asyncio
from typing import Optional, Set
from model_factory import get_model
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.agents.middleware.human_in_the_loop import (
    HITLResponse,
    ApproveDecision,
    EditDecision,
    RejectDecision
)
from langgraph.types import Command
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

load_dotenv(override=True)

async def run_interrupt_test():
    """
    示例 1: 基础中断功能 (封装版)
    在工具调用前中断，让用户确认是否继续执行
    """
    print("\n" + "="*80)
    print("📚 示例 1: interrupt_on 使用")
    print("="*80)
    print("\n功能：在工具调用前暂停，等待人工确认\n")

    # 创建 LLM 和工具
    llm = get_model()
    search_tool = TavilySearch(max_results=2)

    # 创建 Agent，设置在 "tools" 节点中断
    agent = create_deep_agent(
        model=llm,
        tools=[search_tool],
        backend=FilesystemBackend(root_dir="./workspace",virtual_mode=True),
        checkpointer=InMemorySaver(),  # 必需！用于支持中断和恢复
        interrupt_on={"tavily_search": True},  # 在特定工具调用时中断
    )

    # 定义任务
    task = "搜索 'Python 异步编程' 的最新信息，并创建一个总结文件"

    # 配置会话 ID
    # 为了避免之前的状态干扰，我们使用一个新的 thread_id
    config = {"configurable": {"thread_id": "demo_basic_refactored_v1"}}

    print(f"📋 任务: {task}\n")
    print("🚀 开始执行...\n")

    # 追踪已打印的消息数量，避免重复打印
    message_history_len = 0

    # --- 第一次执行 ---
    print("【第一次执行 - 预期会中断】")
    async for event in agent.astream({"messages": [("user", task)]}, config=config):
        if "messages" in event:
            current_messages = event["messages"]
            if len(current_messages) > message_history_len:
                # 打印新增的消息
                for i in range(message_history_len, len(current_messages)):
                    msg = current_messages[i]
                    if msg.type == "ai":
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"🔧 AI 决定调用工具: {msg.tool_calls[0]['name']}")
                            print(f"   参数: {msg.tool_calls[0]['args']}")
                        elif msg.content:
                            print(f"💬 AI: {msg.content}")
                    elif msg.type == "tool":
                        print(f"✅ 工具输出: {msg.content[:100]}..." if len(msg.content) > 100 else f"✅ 工具输出: {msg.content}")

                message_history_len = len(current_messages)

    # 检查是否中断
    # 使用 aget_state (async) 获取状态
    state = await agent.aget_state(config)
    print(f"\n⏸️  执行状态: {state.next}")

    if state.tasks:
        print(f"\n--- 🛑 执行已暂停 (HITL Middleware) ---")
        print(f"下一步骤 (Next): {state.next}")

        last_message = state.values["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            print(f"\n[待审批操作]:")
            print(f"  - 工具: {tool_call['name']}")
            print(f"  - 参数: {tool_call['args']}")

            # === 人工介入 ===
            approval = input("\n[管理员]: 是否批准执行此操作? (y/n/e[编辑]): ")

            if approval.lower() == 'y':
                print("\n[系统]: 操作已批准，继续执行...")

                hitl_response = HITLResponse(
                    decisions=[ApproveDecision(type="approve")]
                )

                # === 恢复执行 ===
                # 使用 Command(resume=...)
                async for event in agent.astream(
                    Command(resume=hitl_response),
                    config=config,
                    stream_mode="values"
                ):
                    if "messages" in event:
                        current_messages = event["messages"]
                        if len(current_messages) > message_history_len:
                            for i in range(message_history_len, len(current_messages)):
                                msg = current_messages[i]

                                # 优化打印逻辑，清晰展示 AI 回复
                                if msg.type == "tool":
                                    print(f"\n[工具输出]:\n{msg.content[:300]}..." if len(msg.content) > 300 else f"\n[工具输出]:\n{msg.content}")
                                elif msg.type == "ai":
                                    if msg.content:
                                        print(f"\n[AI 回复]:\n{msg.content}\n")
                                    elif msg.tool_calls:
                                        print(f"\n🔧 AI 决定调用工具: {msg.tool_calls[0]['name']}")
                                        print(f"   参数: {msg.tool_calls[0]['args']}")

                            message_history_len = len(current_messages)

            else:
                print("\n[系统]: 操作被拒绝或您选择了其他选项 (本演示仅处理 'y')。")
    else:
        print("流程已完成，没有触发中断。")
        if state.values.get("messages"):
            last_msg = state.values["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"\n[最终回复]: {last_msg.content}")

if __name__ == "__main__":
    try:
        asyncio.run(run_interrupt_test())
       # await run_interrupt_test()
    except KeyboardInterrupt:
        print("\n程序已停止")
