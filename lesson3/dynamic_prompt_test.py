#%%
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware import dynamic_prompt
from typing import TypedDict
from model_factory import get_model

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

# 2. 定义上下文结构
class Context(TypedDict):
    user_role: str  # 用户角色

# 3. 动态提示函数
@dynamic_prompt
def role_based_prompt(request:ModelRequest):
    """根据用户角色生成不同提示词"""
    user_role = request.runtime.context.get("user_role", "user")

    if user_role == "expert":
        return "你是一个专业气象分析师，提供详细数据"
    elif user_role == "beginner":
        return "你是一个友善的导游，用简单语言解释"
    else:
        return "你是一个简洁的天气助手"

# 4. 创建动态 Agent
agent_dynamic = create_agent(
    model=get_model("qwen2.5:7b", "ollama"),
    tools=[get_weather],
    middleware=[role_based_prompt],  # 注入动态提示
    context_schema=Context
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