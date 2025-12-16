from langchain_core.tools import tool
from langchain.agents import create_agent
from model_factory import get_model

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model = model,
    tools = [add_numbers],
    system_prompt = "你是一个能够使用加法工具的AI助手。",
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "请帮我计算 15 加 27 等于多少？"}
        ]
    }
)

print("Agent Response:", result)