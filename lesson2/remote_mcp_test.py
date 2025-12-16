from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from model_factory import get_model
import os
import asyncio

async def main():
    mcp_config = {
        # 本地 Python MCP 服务器
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["mcp_server.py"]
        },
        # 高德地图 MCP 服务器
        "amap-maps": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@amap/amap-maps-mcp-server"],
            "env": {
                "AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY"),
            }
        }
    }

    mcp_client = MultiServerMCPClient(mcp_config)
    print("正在连接 MCP 服务器...")

    tools = await mcp_client.get_tools()
    print(f"成功加载 {len(tools)} 个工具: {[t.name for t in tools]}")

    model = get_model("qwen2.5:7b", "ollama")

    agent = create_agent(
        model,
        tools=tools,
        system_prompt="你是一个多功能助手, 可以提供地图查询和数学计算服务。",
    )

    print("\n --- 测试 MCP 工具 --- \n")

    query = "帮我查询一下杭州今天(2025-12-12)的天气, 并计算当天最大温差。"

    inputs = {
                "messages": [
                    HumanMessage(content=query)
                ]
            }

    async for chunk in agent.astream(inputs, stream_mode="values"):
        last_msg = chunk["messages"][-1]
        print(f"\n[{type(last_msg).__name__}]:")
        print(last_msg.content)

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            print(f">>> 调用工具详情: {last_msg.tool_calls}")

if __name__ == "__main__":
    asyncio.run(main())