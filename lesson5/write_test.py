from model_factory import get_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from deepagents import create_deep_agent
from langchain_community.tools import WriteFileTool

deepseek_model = get_model('deepseek-chat', 'deepseek', temperature=0, timeout=300)
ollama_model = get_model('qwen2.5:7b', 'ollama', temperature=0, timeout=300)
google_model = get_model('gemini-2.5-flash', 'google-genai', temperature=0.7, timeout=30)

tavily = TavilySearch(max_results = 3)
file_tool = WriteFileTool(base_path="/home/neko/project/pyProject/langChainLearning/lesson5")

research_instructions = """
你是一位记录员, 每次会把我给你的内容记录到一个本地文件中.
## 可用工具
### `写入本地文件`
使用此功能将研究报告保存到本地文件。文件名使用当时的时间戳, 格式 txt
件中。
- 文件放在路径/home/neko/project/pyProject/langChainLearning/lesson5 下
- 写完文件后, 请告诉我文件保存的位置
"""

agent = create_deep_agent(
    name = "ResearchAgent",
    model = deepseek_model,
    tools = [tavily, file_tool],
    system_prompt = research_instructions,
    checkpointer = InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

message = {
    "messages": [
        {
            "role": "user",
            "content": "1. 今天天气怎么样?\n2. 帮我把这个问题记录到本地文件中."
        }
    ]
}

result = agent.invoke(message, config = config)
print(result["messages"])