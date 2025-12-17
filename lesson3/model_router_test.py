#%%
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Callable,TypedDict
from model_factory import get_model

# 定义上下文结构
class Context(TypedDict):
    user_role: str  # 用户角色

# 1. 定义两个模型实例
small_model = get_model("qwen3:0.6b", "ollama")
large_model = get_model("qwen2.5:7b", "ollama")

hard_keywords = ("证明", "推导", "严谨", "规划","复杂", "多步骤", "chain of thought",
                     "step-by-step", "reason step by step", "数学", "逻辑证明", "约束求解")

# 2. 定义动态模型切换的中间件
@wrap_model_call
async def dynamic_model_router(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    根据对话上下文动态切换模型
    """
    # 获取当前对话的状态（例如消息列表）
    state = request.state
    print(f"--- [Middleware] 当前对话状态: {state} ---")
    messages = state.get("messages", [])
    print(f"--- [Middleware] 当前消息数量: {len(messages)} ---")

    # 获取上下文中的用户角色
    print(f"打印运行时上下文: {request.runtime.context}")

    # === 逻辑判断示例 ===
    # 场景 A: 如果对话轮数超过 5 轮，切换到大模型处理复杂上下文
    if len(messages) > 5:
        print(f"--- [Middleware] 检测到长对话 ({len(messages)} msgs)，切换至 GPT-4o ---")
        # 使用 .override() 方法替换本次调用的模型
        request = request.override(model=large_model)

    # 场景 B: 如果用户输入包含特定关键词 (仅作演示，实际可用分类器)
    elif messages and "复杂分析" in messages[-1].content or any(kw.lower() in messages[-1].content for kw in hard_keywords):
         print("--- [Middleware] 检测到复杂任务，切换至 GPT-4o ---")
         request = request.override(model=large_model)

    else:
        print("--- [Middleware] 使用默认小模型 GPT-4o-mini ---")
        # 默认使用 create_agent 初始化时传入的模型（即 small_model）

    # 继续执行调用
    return await handler(request)

# 3. 定义工具（可选）
@tool
def get_weather(city: str):
    """查询天气"""
    return f"{city} 的天气是晴天"

# 4. 创建 Agent 并注入中间件
agent = create_agent(
    model=small_model,  # 默认模型
    tools=[get_weather],
    middleware=[dynamic_model_router],  # <--- 关键：注入动态路由中间件
    context_schema=Context    # 上下文类型
)

# 5. 测试调用
print(">>> User: 你好")
agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    context={"user_role": "expert"}
)

print("\n>>> User: 请进行复杂的市场分析（模拟触发切换）")
agent.invoke(
    {"messages": [{"role": "user", "content": "请进行复杂的市场分析"}]},
    context={"user_role": "expert"}
)

