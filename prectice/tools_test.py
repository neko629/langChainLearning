from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from openrouter_test import model as gpt_model


@tool("get_current_weather", description="获取指定城市的当前天气信息。")
def get_current_weather(city: str) -> str:
    """
    查询当天天气信息的模拟实现。
    """

    return f"{city}的当前天气是晴朗，温度25摄氏度，湿度60%。"


@tool("outfit_helper", description="根据天气情况推荐适合的穿搭。")
def get_outfit_advice(weather_desc: str) -> str:
    """
    根据天气信息给出穿衣推荐
    :param weather_desc:
    :return:
    """
    admin_input = input("请人工介入穿搭建议环节, y/n/e:")
    if admin_input == "y":
        return f"{weather_desc}, 建议穿着轻便的夏装，如短袖衬衫和短裤。"
    elif admin_input == "n":
        return "穿搭建议服务暂时不可用"
    elif admin_input == "e":
        return input("请输入人工建议:")
    return f"{weather_desc}, 建议穿着轻便的夏装，如短袖衬衫和短裤。"





hmtl_mw = HumanInTheLoopMiddleware(
    interrupt_on={"get_outfit_advice": True},
    description_prefix="需要人工批准"
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=gpt_model,
    tools=[get_current_weather, get_outfit_advice],
    system_prompt="""
    你是一个生活助手, 如果需要查询天气, 请使用 get_current_weather 工具, 然后根据天气情况使用 outfit_helper 工具推荐穿搭。
    如果工具返回不可用, 请如实告知用户
    """,
    middleware=[hmtl_mw],
    checkpointer=checkpointer
)

query = "今天杭州天气怎么样"

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": query}]
    },
    config={"configurable": {"thread_id": "1"}}
)

print(result)
