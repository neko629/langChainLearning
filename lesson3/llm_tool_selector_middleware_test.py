from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware
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

# ==================== 2. 定义多个工具（模拟大量工具场景）====================
@tool
def search_weather(city: str) -> str:
    """查询指定城市的天气信息"""
    logger.info(f"search_weather 被调用: {city}")
    return f"{city}的天气：晴天，温度25°C，湿度60%"

@tool
def search_news(topic: str) -> str:
    """搜索指定主题的最新新闻"""
    logger.info(f"search_news 被调用: {topic}")
    return f"关于'{topic}'的最新新闻：今日头条新闻内容..."

@tool
def calculate_math(expression: str) -> str:
    """计算数学表达式的结果"""
    logger.info(f"calculate_math 被调用: {expression}")
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except:
        return "计算错误"

@tool
def translate_text(text: str, target_lang: str) -> str:
    """将文本翻译成目标语言"""
    logger.info(f"translate_text 被调用: {text} -> {target_lang}")
    return f"翻译结果: [模拟翻译到{target_lang}]"

@tool
def search_database(query: str) -> str:
    """在数据库中搜索信息"""
    logger.info(f"search_database 被调用: {query}")
    return f"数据库搜索结果: 找到3条关于'{query}'的记录"

@tool
def send_email(recipient: str, subject: str) -> str:
    """发送电子邮件"""
    logger.info(f"send_email 被调用: {recipient}, {subject}")
    return f"邮件已发送给 {recipient}"

@tool
def get_stock_price(symbol: str) -> str:
    """获取股票价格"""
    logger.info(f"get_stock_price 被调用: {symbol}")
    return f"股票 {symbol} 当前价格: $150.25"

@tool
def book_meeting(date: str, time: str) -> str:
    """预订会议室"""
    logger.info(f"book_meeting 被调用: {date} {time}")
    return f"会议室已预订: {date} {time}"

# 所有工具列表（模拟拥有大量工具的场景）
all_tools = [
    search_weather,
    search_news,
    calculate_math,
    translate_text,
    search_database,
    send_email,
    get_stock_price,
    book_meeting,
]

class UserContext(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")

tool_selector_middleware = LLMToolSelectorMiddleware(
    model = get_model("qwen3:0.6b", "ollama"),
    max_tools = 3,  # 每次只选择3个最相关的工具
    always_include = ["calculate_math"],
    system_prompt="分析用户查询，选择最相关的工具。优先选择直接相关的工具。"
)

agent = create_agent(
    model=get_model("qwen2.5:7b", "ollama"),
    tools=all_tools,  # 提供所有8个工具
    middleware=[
        tool_selector_middleware,  # 添加工具选择中间件
    ],
    context_schema=UserContext,
    debug=True,  # 开启调试模式以观察工具选择过程
)

def run_tool_selector_test():
    """
    测试 LLMToolSelectorMiddleware 的智能工具选择功能

    场景：从8个工具中智能选择最相关的3个工具
    """
    logger.info("开始 LLMToolSelectorMiddleware 测试")
    logger.info(f"配置: 总共 {len(all_tools)} 个工具，最多选择 3 个，始终包含 calculate_math")

    test_queries = [
        "北京今天的天气怎么样？",
        "帮我计算 123 + 456 的结果",
        "查询苹果公司的股票价格",
        "搜索关于人工智能的最新新闻",
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info("\n" + "="*60)
        logger.info(f"测试场景 {i}: {query}")
        logger.info("="*60)

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                context=UserContext(user_id="user_selector_test"),
                config=ensure_config({"configurable": {"thread_id": f"session_selector_{i:03d}"}})
            )

            final_message = result["messages"][-1]
            logger.info(f"✅ 场景 {i} 完成")
            logger.info(f"响应摘要: {final_message.content[:80]}...")

        except Exception as e:
            logger.error(f"❌ 场景 {i} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 输出说明
    logger.info("\n" + "="*60)
    logger.info("测试完成")
    logger.info("="*60)

    print("\n" + "="*60)
    print("LLMToolSelectorMiddleware 工作原理说明")
    print("="*60)
    print("1. Agent 配置了 8 个不同功能的工具")
    print("2. 中间件使用 LLM 分析用户查询")
    print("3. 从 8 个工具中智能选择最相关的 3 个")
    print("4. calculate_math 工具始终被包含（always_include）")
    print("5. 主模型只能看到被选中的工具")
    print("6. 这样可以减少 token 消耗，提高响应质量")
    print("="*60 + "\n")

    print("\n💡 优势：")
    print("- Token 节省：只传递相关工具描述，减少约 60-70% 的工具相关 token")
    print("- 准确性提升：主模型更容易选择正确的工具")
    print("- 成本降低：减少 API 调用成本")
    print("- 可扩展性：支持数十甚至上百个工具的场景\n")

# ==================== 7. 运行测试 ====================
run_tool_selector_test()