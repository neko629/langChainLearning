from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ensure_config
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
import random
from model_factory import get_model

# ==================== 1. 配置日志 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ==================== 2. 定义工具（模拟可能失败的工具）====================
# 全局计数器，用于模拟间歇性故障
call_counts = {}

@tool
def unreliable_api_call(query: str) -> str:
    """
    模拟不稳定的 API 调用
    前2次调用会失败，第3次成功
    """
    if 'unreliable_api_call' not in call_counts:
        call_counts['unreliable_api_call'] = 0

    call_counts['unreliable_api_call'] += 1
    attempt = call_counts['unreliable_api_call']

    logger.info(f"unreliable_api_call 第 {attempt} 次调用: {query}")

    # 前2次调用失败
    if attempt <= 2:
        logger.warning(f"模拟 API 调用失败（第 {attempt} 次尝试）")
        raise ConnectionError(f"API 连接失败（尝试 {attempt}/3）")

    # 第3次成功
    logger.info(f"✅ API 调用成功（第 {attempt} 次尝试）")
    return f"API 查询成功: '{query}' 的结果数据"

@tool
def stable_tool(data: str) -> str:
    """稳定的工具，总是成功"""
    logger.info(f"stable_tool 被调用: {data}")
    return f"处理完成: {data}"

@tool
def random_failure_tool(input_text: str) -> str:
    """
    随机失败的工具
    50% 概率失败
    """
    logger.info(f"random_failure_tool 被调用: {input_text}")

    if random.random() < 0.5:
        logger.warning("模拟随机失败")
        raise RuntimeError("随机错误：服务暂时不可用")

    logger.info("✅ 随机工具调用成功")
    return f"随机工具处理结果: {input_text}"

tools = [unreliable_api_call, stable_tool, random_failure_tool]

# ==================== 3. 定义上下文 ====================
class UserContext(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")

retry_middleware = ToolRetryMiddleware(
    max_retries = 3,
    tools = ["unreliable_api_call", "random_failure_tool"],
    retry_on = (ConnectionError, RuntimeError),
    on_failure = "return_message",
    backoff_factor = 1.5, # 指数退避因子, 每次重试等待时间增加1.5倍
    initial_delay = 0.5,
    max_delay = 5.0,
    jitter = True,
)

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model,
    tools = tools,
    middleware = [retry_middleware],
    context_schema = UserContext,
    debug = True
)

def run_retry_test():
    """
    测试 ToolRetryMiddleware 的自动重试功能

    场景：测试不稳定工具的自动重试机制
    """
    logger.info("开始 ToolRetryMiddleware 测试")
    logger.info("配置: max_retries=3, 对 unreliable_api_call 和 random_failure_tool 启用重试")

    # 重置计数器
    call_counts.clear()

    # 测试场景1: 不稳定的 API 调用（前2次失败，第3次成功）
    logger.info("\n" + "="*60)
    logger.info("场景1: 测试不稳定的 API 调用（应该在重试后成功）")
    logger.info("="*60)

    query1 = "请调用 unreliable_api_call 查询用户数据"
    logger.info(f"查询: {query1}")

    try:
        result1 = agent.invoke(
            {"messages": [HumanMessage(content=query1)]},
            context=UserContext(user_id="user_retry_test"),
            config=ensure_config({"configurable": {"thread_id": "session_retry_001"}})
        )

        final_message = result1["messages"][-1]
        logger.info(f"✅ 场景1完成")
        logger.info(f"响应: {final_message.content[:100]}...")

    except Exception as e:
        logger.error(f"❌ 场景1失败: {e}")

    # 测试场景2: 稳定工具（不需要重试）
    logger.info("\n" + "="*60)
    logger.info("场景2: 测试稳定工具（不需要重试）")
    logger.info("="*60)

    query2 = "请使用 stable_tool 处理数据"
    logger.info(f"查询: {query2}")

    try:
        result2 = agent.invoke(
            {"messages": [HumanMessage(content=query2)]},
            context=UserContext(user_id="user_retry_test"),
            config=ensure_config({"configurable": {"thread_id": "session_retry_002"}})
        )

        final_message = result2["messages"][-1]
        logger.info(f"✅ 场景2完成")
        logger.info(f"响应: {final_message.content[:100]}...")

    except Exception as e:
        logger.error(f"❌ 场景2失败: {e}")

    # 输出说明
    logger.info("\n" + "="*60)
    logger.info("测试完成")
    logger.info("="*60)

    print("\n" + "="*60)
    print("ToolRetryMiddleware 工作原理说明")
    print("="*60)
    print("1. unreliable_api_call 工具前2次调用失败")
    print("2. 中间件自动捕获 ConnectionError 异常")
    print("3. 使用指数退避策略等待后重试")
    print("4. 第3次调用成功，返回结果")
    print("5. stable_tool 工具始终成功，不需要重试")
    print("6. 重试机制对业务逻辑完全透明")
    print("="*60 + "\n")

    print("\n💡 重试策略：")
    print("- 第1次重试延迟: 0.5秒 × 1.5^0 = 0.5秒")
    print("- 第2次重试延迟: 0.5秒 × 1.5^1 = 0.75秒")
    print("- 第3次重试延迟: 0.5秒 × 1.5^2 = 1.125秒")
    print("- 添加随机抖动避免雷鸣群效应")
    print("\n🎯 适用场景：")
    print("- 网络请求不稳定")
    print("- 外部 API 限流")
    print("- 数据库连接超时")
    print("- 临时性服务故障\n")

# ==================== 7. 运行测试 ====================
run_retry_test()