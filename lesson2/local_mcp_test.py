from langchain_mcp_adapters.client import MultiServerMCPClient   # 导入 MCP 客户端
import os
from langchain_core.tools import tool
from langchain.agents import create_agent
import asyncio
from model_factory import get_model

mcp_server_path = os.path.join("mcp_server.py")
print(mcp_server_path)

# 初始化 mcp 客户端, 链接本地 MCP 服务器
mcp_client = MultiServerMCPClient(
    {
        # 本地 python mcp 服务器 (stdio 传输)
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_server_path],
        },
        # 只能添加在运行中的服务器
    }
)

# 加载 mcp 工具
try:
    mcp_tools = mcp_client.get_tools()
    print(f"成功加载 {len(mcp_tools)} 个 MCP 工具: {[t.name for t in mcp_tools]}")
except Exception as e:
    print("加载 MCP 工具失败:", e)
    mcp_tools = []

# 天气查询工具
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

# 合并所有工具
all_tools = [get_weather] + mcp_tools

model = get_model("qwen2.5:7b", "ollama")

agent = create_agent(
    model,
    tools = all_tools,
    system_prompt = "你是一个多功能助手, 可以提供天气信息和数学计算服务。",
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "查询一下北京和上海气温，并且计算一下北京的温度比上海低多少度？"}
    ]
})

print("Agent Response:", response['messages'][-1].content)

