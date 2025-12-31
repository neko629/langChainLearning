from model_factory import get_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from deepagents import create_deep_agent

deepseek_model = get_model('deepseek-chat', 'deepseek', temperature=0, timeout=300)
ollama_model = get_model('qwen2.5:7b', 'ollama', temperature=0, timeout=300)
google_model = get_model('gemini-2.5-flash', 'google-genai', temperature=0.7, timeout=30)

tavily = TavilySearch(max_results = 3)

research_instructions = """
您是一位资深的研究人员。您的工作是进行深入的研究，然后撰写一份精美的报告。
您可以通过互联网搜索引擎作为主要的信息收集工具。
## 可用工具
### `互联网搜索`
使用此功能针对给定的查询进行互联网搜索。您可以指定要返回的最大结果数量、主题以及是否包含原始内容。
### `写入本地文件`
使用此功能将研究报告保存到本地文件。当您完成研究并生成报告后，请使用此工具将完整的报告内容保存到文
件中。
- 文件放在路径/home/neko/project/pyProject/langChainLearning/lesson5 下, 格式使用 .md 格式（Markdown），例如 "research_report.md" 或 "./reports/报告名
称.md"
- 请确保报告内容完整、结构清晰，包含所有章节和引用来源
## 工作流程
在进行研究时：
1. 首先将研究任务分解为清晰的步骤
2. 使用互联网搜索来收集全面的信息
3. 将信息整合成一份结构清晰的报告
4. **重要**：完成报告后，务必使用 `写入文件工具` 工具将完整报告保存到本地文件, 并告诉用户文件保存的位置
5. 务必引用你的资料来源
**注意**：请确保在完成研究后，将完整的报告内容保存到文件中，这样用户可以方便地查看和保存报告, 文件需要是中文的, 如果你查到的内容不是中文, 请翻译成中文。
"""

agent = create_deep_agent(
    name = "ResearchAgent",
    model = deepseek_model,
    tools = [tavily],
    system_prompt = research_instructions,
    checkpointer = InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

message = {
    "messages": [
        {
            "role": "user",
            "content": "请帮我写一份介绍deepseek的短文。"
        }
    ]
}

result = agent.invoke(message, config = config)
print(result["messages"])