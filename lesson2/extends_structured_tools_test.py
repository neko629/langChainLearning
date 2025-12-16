import os
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver # 导入内存保存器
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import Type
from model_factory import get_model
import asyncio

# 定义结构化工具的输入参数模型
class OrderQueryInput(BaseModel):
    order_id: str = Field(description = "订单的唯一标识符, 格式为 ORD-XXXXXX")
    need_details: bool = Field(default = False, description = "是否需要返回订单的详细信息")

class OrderQueryTool(StructuredTool):

    name: str = "query_order_tool"
    description: str = "用于查询订单状态和明细的工具。输入订单ID, 返回订单状态和可选的详细信息。"
    args_schema: Type[BaseModel] = OrderQueryInput
    return_direct: bool = False # 直接返回结果, 不经过模型再次处理

    def _run(self, order_id: str, need_details: bool = False) -> dict:
        order_db = {
            "ORD-12345": {
                "status": "已发货",
                "price": 150,
                "name": "智能手表",
                "express_number": "EXP123456789"
            },
            "ORD-67890": {
                "status": "处理中",
                "price": 299,
                "name": "无线耳机",
                "express_number": "EXP987654321"
            }
        }

        if order_id not in order_db:
            return {"error": f"未找到订单 {order_id}。"}

        order_info = order_db[order_id]
        result = {"status": order_info["status"]}

        if need_details:
            result = order_info

        return result

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model,
    tools = [OrderQueryTool()],
    system_prompt = "你是一个订单查询助手, 可以查询订单的状态和明细。",
    checkpointer = InMemorySaver() # 使用内存保存器作为检查点
)

async def run_agent():
    config = {
        "configurable": {
            "thread_id": "query_order_thread_001" # 线程 ID, 用于区分不同的对话线程
        },
        "recursion_limit": 15 # 设置递归调用的最大深度, 防止无限递归
    }

    query = "请帮我查询订单 ORD-12345 的状态, 包括详细信息。"

    async for step in agent.astream(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        config = config,
        stream_mode = "values"
    ):
        message = step["messages"][-1]
        message.pretty_print()  # 美化打印输出每一步的消息
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(run_agent())







