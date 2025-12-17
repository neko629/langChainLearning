#%%
# ==================== ModelCallLimitMiddleware 完整实现 ====================

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ensure_config
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from model_factory import get_model

load_dotenv(override=True)

# ==================== 1. 定义工具 ====================
@tool
def complex_calculation(x: int) -> int:
    """执行复杂计算"""
    return x * 2

@tool
def get_weather(city: str) -> str:
    """获取天气信息"""
    return f"{city}的天气：晴天，温度25°C"

tools = [complex_calculation, get_weather]

# ==================== 2. 定义上下文 ====================
class UserContext(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")

# ==================== 3. 配置中间件 ====================
limit_middleware = ModelCallLimitMiddleware(
    run_limit=3,  # 每次运行最多调用模型3次
    exit_behavior='error'  # 超限时抛出异常
)

# ==================== 4. 创建 Agent ====================
agent = create_agent(
    model=get_model("qwen2.5:7b", "ollama"),
    tools=tools,
    middleware=[limit_middleware],
    context_schema=UserContext,
    debug=False,  # 关闭调试模式以减少输出
)

# ==================== 5. 执行测试 ====================
def run_limit_test():
    """测试 ModelCallLimitMiddleware 触发逻辑"""

    # 设计一个需要多次模型调用的任务
    query = """
    请按照以下步骤执行：
    1. 计算 5 的两倍
    2. 用第一步的结果再计算两倍
    3. 用第二步的结果再计算两倍
    4. 用第三步的结果再计算两倍
    5. 最后告诉我北京的天气

    请一步一步执行，每次只做一个计算。
    """

    print("=" * 60)
    print("ModelCallLimitMiddleware 测试")
    print("=" * 60)
    print(f"\n【输入】\n{query.strip()}\n")

    model_call_count = 0
    limit_triggered = False
    final_output = None

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            context=UserContext(user_id="user_limit_test"),
            config=ensure_config({"configurable": {"thread_id": "thread_limit_001"}}),
            stream_mode="updates"
        ):
            if isinstance(chunk, dict):
                for key, value in chunk.items():
                    # 统计模型调用
                    if "model" in str(key).lower():
                        model_call_count += 1

                    # 检测中间件触发
                    if "ModelCallLimitMiddleware" in str(key):
                        limit_triggered = True

                    # 获取最终输出
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages and hasattr(messages[-1], 'content'):
                            final_output = messages[-1].content

        print(f"【输出】\n{final_output}\n")

    except Exception as e:
        print(f"【输出】\n执行被中断: {str(e)}\n")

        if "limit" in str(e).lower() or "exceeded" in str(e).lower():
            limit_triggered = True

    # 输出触发结果
    print("=" * 60)
    print(f"模型调用次数: {model_call_count}")
    print(f"中间件触发: {'✅ 是 (达到 run_limit=3 限制)' if limit_triggered else '❌ 否'}")
    print("=" * 60)

# ==================== 6. 运行测试 ====================
run_limit_test()
