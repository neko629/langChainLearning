from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ensure_config
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
from model_factory import get_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv(override=True)

@tool
def search_database(query: str) -> str:
    """搜索数据库并返回大量结果"""
    # 每次返回约 1000 个字符（约 250 tokens）
    result = f"搜索 '{query}' 的结果：\n"
    result += "\n".join([f"记录 {i}: 这是关于 {query} 的详细信息，包含大量文本内容..." * 5 for i in range(10)])
    logger.info(f"search_database 被调用，查询: {query}，返回约 {len(result)} 字符")
    return result

@tool
def analyze_data(data_id: str) -> str:
    """分析数据并返回详细报告"""
    # 每次返回约 1000 个字符（约 250 tokens）
    result = f"数据 {data_id} 的分析报告：\n"
    result += "详细分析内容包括统计数据、趋势分析、异常检测等..." * 20
    logger.info(f"analyze_data 被调用，数据ID: {data_id}，返回约 {len(result)} 字符")
    return result

@tool
def generate_report(topic: str) -> str:
    """生成报告"""
    result = f"关于 '{topic}' 的报告：\n"
    result += "报告内容包括背景介绍、现状分析、未来展望等..." * 15
    logger.info(f"generate_report 被调用，主题: {topic}，返回约 {len(result)} 字符")
    return result

tools = [search_database, analyze_data, generate_report]

class UserContext(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")

custom_context_middleware = ContextEditingMiddleware(
    edits = [
        ClearToolUsesEdit(
            trigger = 100,
            keep = 1,
            clear_at_least = 0,
            clear_tool_inputs = False, # 保留工具输入以便后续参考
            exclude_tools = ["generate_report"], # 保留报告生成工具的使用记录
            placeholder = "[已清理过多的工具调用记录以优化上下文]",
        )
    ],
    token_count_method = "approximate" # 使用近似方法计算 token 数
)

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model,
    tools = tools,
    middleware = [custom_context_middleware],
    context_schema = UserContext,
    checkpointer = MemorySaver(), # 启用内存检查点保存对话历史, 为什么要保存: 方便调试和后续分析
    debug = True,
)

def run_context_editing_test():
    """测试 ContextEditingMiddleware 的触发逻辑"""
    logger.info("开始 ContextEditingMiddleware 测试")
    logger.info("配置: trigger=800 tokens, keep=1, exclude_tools=['generate_report']")
    logger.info("策略: 在同一线程中执行多次查询，累积消息历史")

    config = ensure_config({"configurable": {"thread_id": "context_editing_test_001"}})
    context = UserContext(user_id = "user_context_editing")

    queries = [
        "请搜索数据库中关于 'AI技术' 的信息",
        "请分析数据 'dataset_001'",
        "请搜索数据库中关于 '机器学习' 的信息",
        "请分析数据 'dataset_002'",
        "请生成关于 '人工智能发展趋势' 的报告",
    ]

    middleware_triggered = False # 标记中间件是否被触发

    for i, query in enumerate(queries):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"第 {i} 次查询: {query}")
        logger.info(f"{'=' * 60}")

        try:
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content = query)]
                },
                context = context,
                config = config
            )

            # 检查消息历史
            messages = result.get("messages", [])
            logger.info(f"当前消息历史长度: {len(messages)} 条")

            # 检查中间件触发情况
            cleared_count = sum(
                1 for msg in messages
                if hasattr(msg, 'response_metadata')
                and msg.response_metadata.get("context_editing", {}).get("clered")
            )

            if cleared_count > 0:
                middleware_triggered = True
                logger.info(f"ContextEditingMiddleware 被触发，清理了 {cleared_count} 条消息记录")
        except Exception as e:
            logger.error(f"执行过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info("测试完成")
    logger.info(
        f"中间件触发: {'✅ 是 - 旧工具结果已被清理' if middleware_triggered else '❌ 否 - 未达到触发阈值'}")
    logger.info("=" * 60)

    # 说明
    print("\n" + "=" * 60)
    print("ContextEditingMiddleware 工作原理说明")
    print("=" * 60)
    print("1. 使用 checkpointer 在同一线程中累积消息历史")
    print("2. 当消息历史超过 800 tokens 时触发清理")
    print("3. 只保留最近的 1 个工具调用结果")
    print("4. 'generate_report' 工具的结果不会被清理（exclude_tools）")
    print("5. 被清理的内容会被替换为 '[已清理以节省空间]'")
    print("6. 每个工具返回约 250 tokens，3-4 次调用后应触发清理")
    print("=" * 60 + "\n")

run_context_editing_test()