# 查看运行流程
import getpass
import operator
from typing import Annotated, List, Union
import os

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent

# 引入 UI 库
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from model_factory import get_model

# 初始化控制台
console = Console()

# 定义天气查询工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": "晴朗，气温25°C",
        "上海": "多云，气温28°C",
        "广州": "小雨，气温30°C"
    }
    return f"{city}的天气是：{weather_data.get(city, '未知')}"

# 定义数学计算工具
@tool
def add(a: float, b: float) -> float:
    """计算两个数的和"""
    return a + b

tools = [get_weather, add]

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model,
    tools = tools
)

def run_demo_with_visualization(user_input: str):
    print("-" * 50)
    console.print(f"[bold yellow]开始任务:[/bold yellow] {user_input}")
    messages = [HumanMessage(content=user_input)]
    step_count = 1 # 步骤计数器

    for event in agent.stream({"messages": messages}, stream_mode = "values"):
        current_message = event["messages"][-1]

        # 如果是人类消息
        if isinstance(current_message, HumanMessage):
            continue # 跳过人类消息, 这是输入

        # 如果是 AI 消息
        if isinstance(current_message, AIMessage):
            # 检查是否有工具调用
            if current_message.tool_calls: # 有调用工具
                for tool_call in current_message.tool_calls:
                    console.print(
                        Panel(
                            Text(
                                f"AI 思考决定: 需要调用外部工具\n"
                                f"工具名称: {tool_call['name']}\n"
                                f"输入参数: {tool_call['args']}", style="bold cyan"
                            ),
                            title=f"步骤 {step_count}: 工具调用",
                            border_style = "cyan"
                        )
                    )
            else:
                # 没有工具调用, 说明是最终回复
                console.print(
                    Panel(
                        Markdown(current_message.content),
                        title = f"Step {step_count}: AI 最终回复",
                        border_style = "green"
                    )
                )
            step_count += 1

        # 如果是工具消息
        elif isinstance(current_message, ToolMessage):
            console.print(
                Panel(
                    Text(f"工具返回结果: {current_message.content}", style="italic white"),
                    title=f"步骤 {step_count}: 执行与观察",
                    border_style = "magenta"
                )
            )
            step_count += 1


run_demo_with_visualization(
            "查询一下北京和上海气温，并且计算一下北京的温度比上海低多少度？")