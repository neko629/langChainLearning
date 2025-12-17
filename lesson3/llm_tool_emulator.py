from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
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
def send_real_email(recipient: str, subject: str, body: str) -> str:
    """
    发送真实邮件（在测试中会被模拟）
    实际生产环境中这会真正发送邮件
    """
    logger.info(f"⚠️ send_real_email 被真实调用: {recipient}")
    # 这里应该是真实的邮件发送逻辑
    return f"真实邮件已发送给 {recipient}，主题: {subject}"

@tool
def charge_credit_card(card_number: str, amount: float) -> str:
    """
    真实扣款（在测试中会被模拟）
    实际生产环境中这会真正扣款
    """
    logger.info(f"⚠️ charge_credit_card 被真实调用: ${amount}")
    # 这里应该是真实的支付逻辑
    return f"已从卡号 {card_number} 扣款 ${amount}"

@tool
def delete_database_record(record_id: str) -> str:
    """
    删除数据库记录（在测试中会被模拟）
    实际生产环境中这会真正删除数据
    """
    logger.info(f"⚠️ delete_database_record 被真实调用: {record_id}")
    # 这里应该是真实的数据库删除逻辑
    return f"记录 {record_id} 已从数据库中删除"

@tool
def safe_query_tool(query: str) -> str:
    """
    安全的查询工具（不会被模拟，真实执行）
    """
    logger.info(f"✅ safe_query_tool 被真实调用: {query}")
    return f"查询结果: 找到关于 '{query}' 的 5 条记录"

tools = [send_real_email, charge_credit_card, delete_database_record, safe_query_tool]

# ==================== 3. 定义上下文 ====================
class UserContext(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")

# ==================== 4. 配置中间件 ====================
# 配置工具模拟中间件：使用 LLM 模拟危险操作，避免真实执行
emulator_middleware = LLMToolEmulator(
    tools=["send_real_email", "charge_credit_card", "delete_database_record"],  # 只模拟这些危险工具
    model=get_model("deepseek-r1:1.5b", "ollama")
)

agent = create_agent(
    model=get_model("qwen2.5:7b", "ollama"),
    tools=tools,
    middleware=[
        emulator_middleware,  # 添加工具模拟中间件
    ],
    context_schema=UserContext,
    debug=True,  # 开启调试模式以观察模拟过程
)

def run_emulator_test():
    """
    测试 LLMToolEmulator 的工具模拟功能

    场景：测试危险操作的模拟执行，确保不会真实执行
    """
    logger.info("开始 LLMToolEmulator 测试")
    logger.info("配置: 模拟 send_real_email, charge_credit_card, delete_database_record")
    logger.info("safe_query_tool 不被模拟，会真实执行")

    test_scenarios = [
        ("场景1: 发送邮件（应该被模拟）", "请发送邮件给 test@example.com，主题是测试邮件"),
        ("场景2: 信用卡扣款（应该被模拟）", "请从卡号 1234-5678-9012-3456 扣款 99.99 美元"),
        ("场景3: 删除数据（应该被模拟）", "请删除数据库中 ID 为 record_123 的记录"),
        ("场景4: 安全查询（应该真实执行）", "请查询用户信息"),
    ]

    for i, (scenario_name, query) in enumerate(test_scenarios, 1):
        logger.info("\n" + "="*60)
        logger.info(scenario_name)
        logger.info("="*60)
        logger.info(f"查询: {query}")

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                context=UserContext(user_id="user_emulator_test"),
                config=ensure_config({"configurable": {"thread_id": f"session_emulator_{i:03d}"}})
            )

            final_message = result["messages"][-1]
            logger.info(f"✅ {scenario_name} 完成")
            logger.info(f"响应摘要: {final_message.content[:80]}...")

        except Exception as e:
            logger.error(f"❌ {scenario_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 输出说明
    logger.info("\n" + "="*60)
    logger.info("测试完成")
    logger.info("="*60)

    print("\n" + "="*60)
    print("LLMToolEmulator 工作原理说明")
    print("="*60)
    print("1. send_real_email, charge_credit_card, delete_database_record 被 LLM 模拟")
    print("2. 这些工具的代码不会被真实执行")
    print("3. LLM 根据工具描述和参数生成合理的模拟结果")
    print("4. safe_query_tool 不在模拟列表中，会真实执行")
    print("5. 日志中可以看到哪些工具被真实调用（⚠️）或模拟（无标记）")
    print("="*60 + "\n")

    print("\n🎯 使用场景：")
    print("- 测试环境：避免执行危险操作（删除、扣款、发送邮件等）")
    print("- 快速原型：无需实现真实工具即可测试 Agent 流程")
    print("- 演示系统：展示功能而不触发真实操作")
    print("- 开发调试：在开发阶段模拟外部 API 调用")
    print("\n💡 最佳实践：")
    print("- 在测试环境中模拟所有危险操作")
    print("- 在生产环境中移除模拟中间件")
    print("- 使用环境变量控制是否启用模拟")
    print("- 模拟结果应该尽可能接近真实结果\n")

# ==================== 7. 运行测试 ====================
run_emulator_test()
