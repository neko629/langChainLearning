#%%
# ==================== ToolCallLimitMiddleware 完整实现 ====================

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ensure_config
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
from model_factory import get_model

# ==================== 1. 配置日志 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ==================== 2. 定义工具 ====================
@tool
def check_server_status(server_id: str) -> str:
    """检查服务器状态"""
    logger.info(f"正在检查服务器 {server_id} 的状态...")
    return f"服务器 {server_id} 运行正常，负载 45%"

@tool
def restart_server(server_id: str) -> str:
    """重启服务器"""
    logger.info(f"正在重启服务器 {server_id}...")
    return f"服务器 {server_id} 已重启"

tools = [check_server_status, restart_server]

# ==================== 3. 定义上下文 ====================
class UserContext(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")

# ==================== 4. 配置中间件 ====================
# 方式1: 限制所有工具的调用次数（全局限制）
global_tool_limiter = ToolCallLimitMiddleware(
    tool_name=None,  # None = 限制所有工具
    run_limit=3,     # 每次运行最多调用 3 次工具
    exit_behavior="continue"  # 超限后阻止工具调用，但继续执行
)

# 方式2: 限制特定工具的调用次数
specific_tool_limiter = ToolCallLimitMiddleware(
    tool_name="check_server_status",  # 只限制 check_server_status 工具
    thread_limit=5,   # 整个线程最多调用 5 次
    run_limit=2,      # 每次运行最多调用 2 次
    exit_behavior="continue"  # 超限后返回错误消息
)

# ==================== 5. 创建 Agent ====================
agent = create_agent(
    model=get_model("qwen2.5:7b", "ollama"),
    tools=tools,
    middleware=[
        specific_tool_limiter,  # 使用特定工具限制器
    ],
    context_schema=UserContext,
    debug=False,
)

# ==================== 6. 执行测试 ====================
def run_tool_limit_test():
    logger.info("开始 ToolCallLimitMiddleware 测试")
    logger.info("配置: check_server_status 工具限制为 run_limit=2")

    # 设计一个会触发多次工具调用的场景
    query = """
    请帮我检查以下服务器的状态：
    1. Server-A
    2. Server-B
    3. Server-C
    4. Server-D

    请逐个检查每台服务器。
    """

    logger.info(f"用户查询: {query.strip()}")

    tool_call_count = 0
    limit_triggered = False

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            context=UserContext(user_id="user_tool_limit"),
            config=ensure_config({"configurable": {"thread_id": "session_tool_limit_001"}}),
            stream_mode="updates"
        ):
            if isinstance(chunk, dict):
                for key, value in chunk.items():
                    # 统计工具调用
                    if "tools" in str(key).lower():
                        tool_call_count += 1

                    # 检测中间件触发
                    if "ToolCallLimitMiddleware" in str(key):
                        limit_triggered = True
                        logger.warning("检测到 ToolCallLimitMiddleware 触发！")

        logger.info("任务完成")

    except Exception as e:
        logger.error(f"捕获到异常: {e}")
        if "limit" in str(e).lower() or "exceeded" in str(e).lower():
            limit_triggered = True
        return str(e)

    # 输出结果
    logger.info("=" * 60)
    logger.info(f"工具调用次数: {tool_call_count}")
    logger.info(f"中间件触发: {'✅ 是' if limit_triggered else '❌ 否'}")
    logger.info("=" * 60)

# ==================== 7. 运行测试 ====================
run_tool_limit_test()