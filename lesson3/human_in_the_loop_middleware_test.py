import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import (
    HITLResponse,
    ApproveDecision,
    EditDecision,
    RejectDecision
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.types import Command
from model_factory import get_model
from langgraph.checkpoint.memory import MemorySaver

# 1. 加载环境变量
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# 2. 定义工具 (Tools)
# ---------------------------------------------------------------------------
class SendEmailSchema(BaseModel):
    recipient: str = Field(description="邮件接收者的邮箱地址")
    subject: str = Field(description="邮件主题")
    body: str = Field(description="邮件正文内容")

@tool(args_schema=SendEmailSchema)
def send_email(recipient: str, subject: str, body: str):
    """模拟发送邮件的工具"""
    print(f"\n======== [SYSTEM ACTION: 正在执行发送邮件] ========")
    print(f"收件人: {recipient}")
    print(f"主题  : {subject}")
    print(f"内容  : {body}")
    print(f"================================================\n")
    return f"邮件已成功发送给 {recipient}"

tools = [send_email]

model = get_model("qwen2.5:7b", "ollama")

system_prompt = """
你是一个专业的行政助手。
当用户请求发送邮件时，你必须直接调用 `send_email` 工具。
不要问任何后续问题，不要要求确认，直接生成工具调用。
"""

hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={"send_email": True},
    description_prefix="需要人工批准才能发送邮件"
)

graph = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[hitl_middleware]
)

def run_interactive_session():
    local_graph = create_agent(
        model = model,
        tools = tools,
        system_prompt = system_prompt,
        middleware = [hitl_middleware],
        checkpointer = MemorySaver()  # 本地运行需要
    )

    thread_id = "demo_thread_middleware_1"
    config = {"configurable": {"thread_id": thread_id}}

    user_input = "帮我给 hr@example.com 发一封邮件，主题是'休假申请'，内容是我下周一想请假一天。"
    print(f"\n[用户]: {user_input}")
    # === 第一步：初始执行 ===
    print("\n[系统]: 开始处理请求...")
    # input 传入用户消息
    # stream_mode="values" 可以让我们看到消息流
    for event in local_graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config = config,
            stream_mode = "values"
    ):
        # 简单打印最后一条消息的内容
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai" and last_msg.tool_calls:
                print(f"[AI 思考]: 决定调用工具 -> {last_msg.tool_calls[0]['name']}")

    # === 第二步：观察 (Observation) ===
    # 中间件应该触发了中断
    snapshot = local_graph.get_state(config)

    print(f"\n--- 🛑 执行已暂停 (HITL Middleware) ---")
    print(f"下一步骤 (Next): {snapshot.next}")
    print(f"任务数量: {len(snapshot.tasks) if snapshot.tasks else 0}")

    # 检查是否有任务（这表示中断发生）
    if snapshot.tasks:
        # 获取最后一条消息
        last_message = snapshot.values["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            print(f"\n[待审批操作]:")
            print(f"  - 工具: {tool_call['name']}")
            print(f"  - 参数: {tool_call['args']}")

            # === 第三步：人工介入 (Human Input) ===
            approval = input("\n[管理员]: 是否批准执行此操作? (y/n/e[编辑]): ")

            if approval.lower() == 'y':
                # === 第四步：恢复执行 (Resume) - 批准 ===
                print("\n[系统]: 操作已批准，继续执行...")

                # 创建 HITLResponse 对象，包含 ApproveDecision
                hitl_response = HITLResponse(
                    decisions = [ApproveDecision(type = "approve")]
                )

                # 使用 Command(resume=hitl_response) 来批准并继续执行
                for event in local_graph.stream(
                        Command(resume = hitl_response),  # 传入 HITLResponse 对象
                        config = config,
                        stream_mode = "values"
                ):
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if last_msg.type == "tool":
                            print(f"[工具输出]: {last_msg.content}")
                        elif last_msg.type == "ai" and last_msg.content:
                            print(f"[AI 回复]: {last_msg.content}")

            elif approval.lower() == 'e':
                # === 编辑工具调用参数 ===
                print("\n[系统]: 编辑模式...")
                print(f"当前参数: {tool_call['args']}")

                # 让用户编辑参数
                new_recipient = input(
                    f"新收件人 (当前: {tool_call['args'].get('recipient', '')}，留空保持不变): ").strip()
                new_subject = input(
                    f"新主题 (当前: {tool_call['args'].get('subject', '')}，留空保持不变): ").strip()
                new_body = input(
                    f"新内容 (当前: {tool_call['args'].get('body', '')}，留空保持不变): ").strip()

                # 构建新的参数
                updated_args = tool_call['args'].copy()
                if new_recipient:
                    updated_args['recipient'] = new_recipient
                if new_subject:
                    updated_args['subject'] = new_subject
                if new_body:
                    updated_args['body'] = new_body

                print(f"\n[系统]: 使用更新后的参数继续执行...")
                print(f"更新后的参数: {updated_args}")

                # 创建 HITLResponse 对象，包含 EditDecision
                # EditDecision 需要 edited_action，包含 name 和 args
                hitl_response = HITLResponse(
                    decisions = [EditDecision(
                        type = "edit",
                        edited_action = {
                            "name": tool_call['name'],
                            "args": updated_args
                        }
                    )]
                )

                # 使用 Command(resume=hitl_response) 来批准并使用新参数
                for event in local_graph.stream(
                        Command(resume = hitl_response),  # 传入包含编辑决策的 HITLResponse
                        config = config,
                        stream_mode = "values"
                ):
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if last_msg.type == "tool":
                            print(f"[工具输出]: {last_msg.content}")
                        elif last_msg.type == "ai" and last_msg.content:
                            print(f"[AI 回复]: {last_msg.content}")

            else:
                # === 拒绝操作 ===
                print("\n[系统]: 操作被拒绝。")

                # 创建 HITLResponse 对象，包含 RejectDecision
                rejection_reason = input("拒绝原因 (可选): ").strip() or "操作被管理员拒绝"

                hitl_response = HITLResponse(
                    decisions = [RejectDecision(
                        type = "reject",
                        message = rejection_reason
                    )]
                )

                # 使用 Command(resume=hitl_response) 来拒绝
                for event in local_graph.stream(
                        Command(resume = hitl_response),  # 传入包含拒绝决策的 HITLResponse
                        config = config,
                        stream_mode = "values"
                ):
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if last_msg.type == "ai" and last_msg.content:
                            print(f"[AI 回复]: {last_msg.content}")
                        elif last_msg.type == "tool":
                            print(f"[工具消息]: {last_msg.content}")

                print("[系统]: 流程已终止。")
        else:
            print("没有检测到待处理的工具调用。")
    else:
        print("流程已完成，没有触发中断。")
        # 打印最终结果
        if snapshot.values.get("messages"):
            last_msg = snapshot.values["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"\n[最终回复]: {last_msg.content}")

run_interactive_session()