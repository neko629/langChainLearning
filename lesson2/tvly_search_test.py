from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults # 导入第三方社区集成 Tavily 搜索工具
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
from model_factory import get_model

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

# 创建 Tavily 搜索工具实例
web_search = TavilySearchResults(
    tavily_api_key = tavily_api_key,
    max_results = 3
)
# 创建模型实例
model = get_model("qwen2.5:7b", "ollama")
# 创建 agent
agent = create_agent(
    model,
    tools = [web_search],
    system_prompt = "你是一个能够使用网络搜索工具的AI助手。",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我搜索一下 2024 年诺贝尔物理学奖的得主。"
            }
        ]
    }
)

print("Agent Response:", result)
