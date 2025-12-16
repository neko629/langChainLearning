from langchain.agents import create_agent
from langchain.tools import tool
from model_factory import get_model
from langchain.agents.middleware import dynamic_prompt
from typing import TypedDict

# 1. 定义一个简单的天气查询工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": "晴朗，气温25°C",
        "上海": "多云，气温28°C",
        "广州": "小雨，气温30°C"
    }
    return f"{city}的天气是：{weather_data.get(city, '未知')}"

model = get_model("qwen2.5:7b", "ollama")

# 2. 静态 system_prompt（固定不变）
agent_static = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=(
        "你是一个天气助手，回答不超过20字。\n"
        "调用工具时，严格按照以下格式：\n"
        "1. 使用 `get_weather(city: str)` 获取天气；\n"
        "2. 仅返回天气结果，不解释过程。"
    )
)

print("=== 静态 System Prompt ===")
response1 = agent_static.invoke({
    "messages": [{"role": "user", "content": "北京天气"}]
})
print(f"AI: {response1}")

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def role_based_prompt(request):
    """根据用户角色动态生成 system_prompt"""
    user_role = request.runtime.context.get("user_role", "user")
    if user_role == "expert":
        return (
            "你是一个专业的天气专家，回答详细且专业。\n"
            "调用工具时，严格按照以下格式：\n"
            "1. 使用 `get_weather(city: str)` 获取天气；\n"
            "2. 提供详细解释和建议。"
        )
    elif user_role == "beginner":
        return (
            "你是一个天气助手，回答简单易懂。\n"
            "调用工具时，严格按照以下格式：\n"
            "1. 使用 `get_weather(city: str)` 获取天气；\n"
            "2. 用简单语言解释天气情况。"
        )
    else:
        return (
            "你是一个天气助手，回答不超过10字。\n"
            "调用工具时，严格按照以下格式：\n"
            "1. 使用 `get_weather(city: str)` 获取天气；\n"
            "2. 仅返回天气结果，不解释过程。"
        )

# 6. 创建动态 Agent
agent_dynamic = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[role_based_prompt],  # 注入动态提示
    context_schema=Context # 指定上下文类型
)

print("\n=== 动态 System Prompt（专家角色）===")
response2 = agent_dynamic.invoke(
    {"messages": [{"role": "user", "content": "北京天气"}]},
    context={"user_role": "expert"}
)
print(f"AI: {response2['messages'][-1].content}")

print("\n=== 动态 System Prompt（新手角色）===")
response3 = agent_dynamic.invoke(
    {"messages": [{"role": "user", "content": "北京天气"}]},
    context={"user_role": "beginner"}
)
print(f"AI: {response3['messages'][-1].content}")

print("\n=== 动态 System Prompt（访客角色）===")
response4 = agent_dynamic.invoke(
    {"messages": [{"role": "user", "content": "北京天气"}]},
    context={"user_role": "visitor"}
)
print(f"AI: {response4['messages'][-1].content}")