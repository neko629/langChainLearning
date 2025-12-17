#%%
# ==================== ModelFallbackMiddleware 完整实现 ====================

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
from model_factory import get_model
from langchain_core.runnables import ensure_config

# ==================== 1. 配置日志 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ==================== 2. 定义工具 ====================
@tool
def calculate_sum(a: int, b: int) -> int:
    """计算两个数的和"""
    logger.info(f"calculate_sum 被调用: {a} + {b}")
    return a + b

@tool
def get_system_info() -> str:
    """获取系统信息"""
    logger.info("get_system_info 被调用")
    return "系统运行正常，CPU使用率: 45%, 内存使用率: 60%"

tools = [calculate_sum, get_system_info]

# ==================== 3. 定义上下文 ====================
class UserContext(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")

# ==================== 4. 配置中间件 ====================
# 配置模型故障转移：主模型 -> 备用模型1 -> 备用模型2
# 注意：这里使用相同的模型作为演示，实际应用中应使用不同的模型
fallback_middleware = ModelFallbackMiddleware(
    get_model("deepseek-r1:1.5b", "ollama"),  # 第一个备用模型
    get_model("qwen3:0.6b", "ollama"),    # 第二个备用模型
)

# ==================== 5. 创建 Agent ====================
agent = create_agent(
    model=get_model("qwen2.5:7b", "ollama"),  # 主模型
    tools=tools,
    middleware=[
        fallback_middleware,  # 添加故障转移中间件
    ],
    context_schema=UserContext,
    debug=False,
)

# ==================== 6. 执行测试 ====================
def run_fallback_test():
    """
    测试 ModelFallbackMiddleware 的故障转移功能

    场景：正常情况下使用主模型，模拟故障时自动切换到备用模型
    """
    logger.info("开始 ModelFallbackMiddleware 测试")
    logger.info("配置: 主模型(qwen2.5) + 2个备用模型")

    # 测试场景1: 正常调用（主模型成功）
    logger.info("\n" + "="*60)
    logger.info("场景1: 正常调用 - 主模型应该成功处理")
    logger.info("="*60)

    query1 = "请计算 15 + 27 的结果"
    logger.info(f"查询: {query1}")

    try:
        result1 = agent.invoke(
            {"messages": [HumanMessage(content=query1)]},
            context=UserContext(user_id="user_fallback_test"),
            config=ensure_config({"configurable": {"thread_id": "session_fallback_001"}})
        )

        final_message = result1["messages"][-1]
        logger.info(f"✅ 场景1成功: {final_message.content[:100]}...")

    except Exception as e:
        logger.error(f"❌ 场景1失败: {e}")

    # 测试场景2: 复杂查询
    logger.info("\n" + "="*60)
    logger.info("场景2: 复杂查询 - 测试模型处理能力")
    logger.info("="*60)

    query2 = "请先获取系统信息，然后计算 100 + 200 的结果，最后总结一下"
    logger.info(f"查询: {query2}")

    try:
        result2 = agent.invoke(
            {"messages": [HumanMessage(content=query2)]},
            context=UserContext(user_id="user_fallback_test"),
            config=ensure_config({"configurable": {"thread_id": "session_fallback_002"}})
        )

        final_message = result2["messages"][-1]
        logger.info(f"✅ 场景2成功: {final_message.content[:100]}...")

    except Exception as e:
        logger.error(f"❌ 场景2失败: {e}")

    # 输出说明
    logger.info("\n" + "="*60)
    logger.info("测试完成")
    logger.info("="*60)

    print("\n" + "="*60)
    print("ModelFallbackMiddleware 工作原理说明")
    print("="*60)
    print("1. 主模型: deepseek-chat (temperature=0.1)")
    print("2. 备用模型1: deepseek-reasoner (temperature=0.3)")
    print("3. 备用模型2: deepseek-chat (temperature=0.5)")
    print("4. 当主模型调用失败时，自动尝试备用模型1")
    print("5. 如果备用模型1也失败，继续尝试备用模型2")
    print("6. 返回第一个成功的模型响应")
    print("7. 实际应用中应配置不同的模型提供商（如 OpenAI, Anthropic 等）")
    print("="*60 + "\n")

    print("\n💡 提示：")
    print("在生产环境中，建议配置不同提供商的模型，例如：")
    print("  主模型: openai:gpt-4o")
    print("  备用1: anthropic:claude-sonnet-4-5-20250929")
    print("  备用2: deepseek:deepseek-chat")
    print("这样可以在某个提供商服务中断时，自动切换到其他提供商。\n")

# ==================== 7. 运行测试 ====================
run_fallback_test()