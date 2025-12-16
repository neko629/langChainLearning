import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  # 短期记忆存储
from langchain_core.messages import trim_messages  # 消息裁剪工具
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model_factory import get_model
import tiktoken

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model=model
)

# 定义一个简单的天气查询工具
@tool
def search_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}天气：晴，25°C"

# ============ 配置裁剪参数 ============
MAX_TOKENS = 60  # 根据模型上下文长度调整，上下文128K的话，一般设置为4000左右
TRIM_STRATEGY = "last"  # 保留最新消息
INCLUDE_SYSTEM = True  # 系统消息不参与裁剪


def count_tokens_with_model(messages: list, model) -> int:
    """使用模型自带的 tokenizer 计算 token 数"""
    total_tokens = 0
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        # 调用模型的 get_num_tokens 方法
        total_tokens += model.get_num_tokens(content)
    return total_tokens

# ============ 创建Agent ============
def create_trimmed_agent():
    """创建 Agent 并配置 InMemorySaver"""
    memory = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[search_weather],
        system_prompt="你是一个简洁的助手，记住用户提到的城市名称",
        checkpointer=memory  # 启用短期记忆
    )

    return agent


def invoke_with_trim(agent, user_input: str, config: dict):
    """
    在调用 Agent 前手动裁剪上下文
    """
    # 1. 获取当前状态（所有历史消息）
    state = agent.get_state(config)
    existing_messages = state.values.get("messages", []) if state else []

    # 计算当前 token 数
    current_tokens = count_tokens_with_model(existing_messages, model)

    # 2. 如果有历史消息，进行裁剪
    if existing_messages:
        print(f"裁剪前消息数: {len(existing_messages)}, Token数: {current_tokens}")

        # 创建适配器函数，只接受消息列表参数
        def token_counter_adapter(messages):
            return count_tokens_with_model(messages, model)

        # 使用 trim_messages 裁剪
        trimmed_messages = trim_messages(
            existing_messages,
            max_tokens=MAX_TOKENS,
            token_counter=token_counter_adapter,  # 使用适配器函数
            strategy=TRIM_STRATEGY,
            include_system=INCLUDE_SYSTEM,
            allow_partial=False,
            start_on="human"
        )
        # 计算裁剪后 token 数
        trimmed_tokens = count_tokens_with_model(trimmed_messages, model)
        print(f"裁剪后消息数: {len(trimmed_messages)}, Token数: {trimmed_tokens}")
    else:
        trimmed_messages = []

    # 3. 构建新输入（裁剪后的消息 + 新消息）
    new_messages = trimmed_messages + [HumanMessage(content=user_input)]

    # 4. 调用 Agent
    response = agent.invoke(
        {
            "messages": new_messages
        },
        config=config
    )

    return response


# ============ 测试裁剪功能 ============
def demo_manual_trim():
    print("=" * 60)
    print("场景：手动 trim_messages + InMemorySaver")
    print("=" * 60)

    agent = create_trimmed_agent()
    config = {"configurable": {"thread_id": "trim_user_001"}}

    # 模拟多轮对话
    conversations = [
        "你好，我叫陈明",
        "查询北京天气",
        "上海呢？",
        "明天北京天气如何？",  # 此时会触发裁剪
        "我是谁？",  # 测试记忆是否保留
    ]

    for i, query in enumerate(conversations, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"用户: {query}")

        # 每次调用前自动裁剪
        response = invoke_with_trim(agent, query, config)

        print(f"AI: {response['messages'][-1].content}")
        print("-" * 40)

if __name__ == "__main__":
    demo_manual_trim()