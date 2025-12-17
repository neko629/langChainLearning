from ipaddress import summarize_address_range

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ensure_config
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import logging
from model_factory import get_model

# 配置日志
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv(override = True)

# 定义工具
@tool
def search_patent(query: str) -> str:
    """搜索专利数据库"""
    return f"专利搜索结果: 找到与 '{query}' 相关的 3 项专利..."

@tool
def analyze_technology(tech_desc: str) -> str:
    """分析技术可行性"""
    return f"技术分析: '{tech_desc}' 的实现可行性评估完成..."

tools = [search_patent, analyze_technology]

# 定义上下文
class UserContext(BaseModel):
    user_id: str = Field(..., description = "用户唯一标识")
    department: str = Field(..., description = "用户所属部门")
    max_history_tokens: Optional[int] = Field(2000, description = "最大历史消息Token数")


model = get_model("qwen2.5:7b", "ollama")
# 配置中间件
summarization_middleware = SummarizationMiddleware(
    model = get_model("deepseek-r1:1.5b", "ollama"),
    trigger = [("tokens", 2000), ("messages", 15)],
    keep = ("messages", 10),
    summary_prompt = "请将以下对话历史进行摘要，保留关键决策点和技术细节：\n\n{messages}\n\n摘要:"
)

# 创建 Agent
agent = create_agent(
    model,
    tools = tools,
    middleware = [summarization_middleware],
    context_schema = UserContext,
    debug = True,
)

# 测试调用
def run_summarization_test():
    logger.info("开始 Summarization Middleware 测试...")

    # 创建长历史对话
    long_history = [HumanMessage(content = f"用户问题 {i + 1}: 如何评估某项技术的专利风险。") for i in range(20)]
    logger.info(f"创建了{len(long_history)}条问题")

    result = agent.invoke(
        {
            "messages": long_history,
        },
        context = UserContext(user_id = "user_123", department = "研发部"),
        config = ensure_config({"configurable": {"thread_id": "summarization_test_001"} })
    )

    result_messages = result.get("messages", [])
    logger.info(f"执行后消息数量: {len(result_messages)}")
    if len(result_messages) < len(long_history):
        logger.info(f"中间件被触发! 压缩了 {len(long_history) - len(result_messages)} 条消息.")

    return result

result = run_summarization_test()
logger.info("test done")