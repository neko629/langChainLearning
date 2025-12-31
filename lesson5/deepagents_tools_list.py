from rich.console import Console
from rich.table import Table
from rich.panel import Panel
RICH_AVAILABLE = True
console = Console()

def print_agent_tools(agent):
    if hasattr(agent, 'nodes') and 'tools' in agent.nodes:
        tools_node = agent.nodes['tools']
        if hasattr(tools_node, 'bound'):
            tool_node = tools_node.bound

            if hasattr(tool_node, 'tools_by_name'):
                tools = tool_node.tools_by_name

                # 3 kinds of tools
                user_tools = []
                filesystem_tools = []
                system_tools = []

                for tool_name, tool in tools.items():
                    tool_info = {
                        'name': tool_name,
                        'description': getattr(tool, 'description', 'No description available')
                    }

                    # Categorize tools
                    if tool_name in ['ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'execute']:
                        filesystem_tools.append(tool_info)
                    elif tool_name in ['write_todos', 'task']:
                        system_tools.append(tool_info)
                    else:
                        user_tools.append(tool_info)
                _print_tools_rich(user_tools, filesystem_tools, system_tools)
            else:
                print("No tools found in the agent.")
        else:
            print("The 'tools' node is not bound.")
    else:
        print("The agent does not have a 'tools' node.")

def _print_tools_rich(user_tools, filesystem_tools, system_tools):
    """使用 Rich 库美化打印工具列表"""
    console.print()

    # 创建表格
    table = Table(title="Agent 加载的工具列表", show_header=True, header_style="bold magenta")
    table.add_column("类别", style="cyan", width=20)
    table.add_column("工具名称", style="green", width=20)
    table.add_column("描述", style="white", width=60)

    # 添加用户工具
    for i, tool in enumerate(user_tools):
        category = "用户工具" if i == 0 else ""
        desc = tool['description'][:80] + "..." if len(tool['description']) > 80 else tool['description']
        table.add_row(category, tool['name'], desc)

    # 添加文件系统工具
    for i, tool in enumerate(filesystem_tools):
        category = "文件系统工具" if i == 0 else ""
        desc = tool['description'][:80] + "..." if len(tool['description']) > 80 else tool['description']
        table.add_row(category, tool['name'], desc)

    # 添加系统工具
    for i, tool in enumerate(system_tools):
        category = "系统工具" if i == 0 else ""
        desc = tool['description'][:80] + "..." if len(tool['description']) > 80 else tool['description']
        table.add_row(category, tool['name'], desc)

    console.print(table)

    # 打印统计
    total = len(user_tools) + len(filesystem_tools) + len(system_tools)
    console.print(Panel(
        f"[bold green]共计 {total} 个工具[/bold green]\n\n"
        f"• 用户工具: {len(user_tools)} 个\n"
        f"• 文件系统工具: {len(filesystem_tools)} 个\n"
        f"• 系统工具: {len(system_tools)} 个",
        title="统计信息",
        border_style="green"
    ))
    console.print()

from model_factory import get_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from deepagents import create_deep_agent

deepseek_model = get_model('deepseek-chat', 'deepseek', temperature=0.7, timeout=30)
ollama_model = get_model('qwen2.5:7b', 'ollama', temperature=0.7, timeout=30)
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
- 文件路径建议使用 .md 格式（Markdown），例如 "research_report.md" 或 "./reports/报告名
称.md"
- 请确保报告内容完整、结构清晰，包含所有章节和引用来源
## 工作流程
在进行研究时：
1. 首先将研究任务分解为清晰的步骤
2. 使用互联网搜索来收集全面的信息
3. 将信息整合成一份结构清晰的报告
4. **重要**：完成报告后，务必使用 `写入本地文件` 工具将完整报告保存到本地文件
5. 务必引用你的资料来源
**注意**：请确保在完成研究后，将完整的报告内容保存到文件中，这样用户可以方便地查看和保存报告。
"""

agent = create_deep_agent(
    name = "ResearchAgent",
    model = ollama_model,
    tools = [tavily],
    system_prompt = research_instructions,
    #checkpointer = InMemorySaver()
)


if __name__ == "__main__":
    print_agent_tools(agent)

