import os
from typing import Annotated
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import re
from dotenv import load_dotenv
from model_factory import get_model

load_dotenv(override = True)


# 定义工具
@tool
def verify_credit_card(card_number: Annotated[str, "信用卡号"]) -> dict:
    """
    验证信用卡卡号有效性(模拟工具)
    注意: 生产环境不应该接受真实卡号
    """
    print(f"verify tool invoked with card_number: {card_number}")

    if (len(card_number) >= 16):
        return {
            "valid": True,
            "message": f"{card_number} 信用卡号有效",
            "type": "Visa"
        }
    return {
        "valid": False,
        "message": f"{card_number} 信用卡号无效",
        "type": "Unknown"
    }


@tool
def process_payment(card_number: str, amount: float) -> str:
    """
    处理支付(模拟工具)
    注意: 生产环境不应该接受真实卡号
    """
    print(f"payment tool invoked with card_number: {card_number}, amount: {amount}")
    return f"已使用信用卡 {card_number} 支付金额 {amount} 元"


@tool
def search_user_history(user_id: str) -> str:
    """
    查询用户历史记录(模拟工具)
    """
    print(f"search tool invoked with user_id: {user_id}")
    return f"用户 {user_id} 的历史记录：购买过电子产品和家用电器。"


tools: list[BaseTool] = [verify_credit_card, process_payment, search_user_history]


# 定义用户上下文
class UserContext(BaseModel):
    user_id: str = Field(..., description = "用户唯一标识")
    department: str = Field(..., description = "用户所属部门")
    security_level: str = Field("normal", description = "用户安全级别")


# 配置 PIIMiddleware
pii_middleware = PIIMiddleware(
    "credit_card",
    detector = r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    strategy = "mask",
    apply_to_input = True,
    apply_to_output = False
)

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model,
    tools = tools,
    middleware = [pii_middleware],
    context_schema = UserContext,
    debug = True,
)


def test_credit_card_masking():
    print("=" * 60)
    print("用户使用信用卡尝试支付")
    print("=" * 60)

    test_query = """
        请帮我验证以下信用卡是否有效：
        我的卡号是 4532-1234-5678-9010，另外备用卡是 4532123456781234。
        请检查这两张卡，然后处理一笔 99.99 美元的支付。
        """

    print(f"user input:\n {test_query}\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content = test_query)
            ],
        },
        context = UserContext(
            user_id = "user_456",
            department = "财务部",
            security_level = "high"
        ),
        config = {
            "configurable": {
                "thread_id": "pii_middleware_test_001"
            }
        }
    )

    print("\nAI final output")
    final_message = result.get("messages", [])[-1]

    if isinstance(final_message, AIMessage):
        print(f"role: {final_message.type}")
        print(f"content: {final_message.content}")

        # tools records
        if hasattr(final_message, "tool_calls") and final_message.tool_calls:
            print(f"\n工具调用记录:")
            for call in final_message.tool_calls:
                print(f"工具: {call.tool_name}, 输入: {call.input}, 输出: {call.output}")

    return result


test_credit_card_masking()

