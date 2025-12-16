from langchain.agents import create_agent
from langchain_core.tools import tool
from model_factory import get_model

@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气信息。"""
    # 这里可以集成实际的天气API调用, 目前返回模拟数据
    weather_data = {
        "北京": "晴, 25°C",
        "上海": "多云, 22°C",
        "广州": "雷阵雨, 28°C"
    }
    return weather_data.get(city, "未找到该城市的天气信息。")

@tool
def calculate(expression: str) -> str:
    """计算给定的数学表达式。"""
    try:
        result = eval(expression)
        return f"计算结果是: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model = model,
    tools = [get_weather, calculate],
    system_prompt = "你是一个多功能助手, 可以提供天气信息和计算服务。",
)

user_queries = [
    "今天北京和上海的天气怎么样？",
    "上海温度是17度, 北京的温度是20度, 两地温差几度？"
]

for query in user_queries:
    print(f"User Query: {query}")
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
    )
    print("Agent Response:", result['messages'][-1].content)
    print("-" * 50)