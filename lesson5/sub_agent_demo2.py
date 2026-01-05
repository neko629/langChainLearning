from deepagents.middleware import subagents
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from model_factory import get_model
from deepagents import create_deep_agent
from langchain_tavily import TavilySearch

console = Console()

async def setup_mcp_tools():
    console.print("[dim]正在连接 Context7 MCP 服务器...[/dim]")
    # 检查 node 环境
    try:
        client = MultiServerMCPClient({
            "context7": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp@latest"],
            }
        })
        # 获取工具
        tools = await client.get_tools()
        console.print(f"[green]成功加载 {len(tools)} 个 MCP 工具[/green]")
        return client, tools
    except Exception as e:
        console.print(f"[red]连接 MCP 失败: {e}[/red]")
        console.print("[yellow]将使用模拟工具继续...[/yellow]")
        return None, []

def get_subagents_config(mcp_tools):
    # 子 Agent 1: 官方文档专家
    doc_tools = mcp_tools if mcp_tools else [TavilySearch(max_results=3)]

    docs_researcher = {
        "name": "DocsResearcher",
        "description": "负责查阅官方文档和技术规范的专家 Agent。",
        "system_prompt": "你是一名专门查阅官方文档的技术专家。请使用工具获取准确的技术细节。不要猜测。",
        "tools": doc_tools,
        "model": "deepseek-chat"
    }

    # 子 Agent 2: 社区生态专家
    community_researcher = {
        "name": "CommunityResearcher",
        "description": "负责搜索社区博客、教程和最佳实践的专家 Agent。",
        "system_prompt": "你是一名关注社区动态的开发者。请搜索博客、论坛和 GitHub 讨论。",
        "tools": [TavilySearch(max_results=3)],
        "model": "deepseek-chat"
    }

    return [docs_researcher, community_researcher]

async def run_auto_subagent_demo():
    console.print(
        Panel.fit("[bold magenta]DeepAgents 自动 SubAgent 中间件演示[/bold magenta]",
                  border_style="magenta"))
    # 初始化 MCP
    mcp_client, mcp_tools = await setup_mcp_tools()

    # 获取子 Agent 配置
    subagents = get_subagents_config(mcp_tools)

    deepseek_model = get_model(
        'deepseek-chat', 'deepseek', temperature=0, timeout=300)

    console.print("[bold cyan]正在创建 Agent (subagents=None)...[/bold cyan]")

    agent = create_deep_agent(
        model = deepseek_model,
        tools = [],
        subagents = subagents,
        system_prompt = """你是一名技术总监。你的任务是协调 DocsResearcher 和 CommunityResearcher 完成调研任务。
                        请根据用户需求，将任务拆解并分发给这两个子 Agent。
                        如果任务允许，请务必并行调用它们以提高效率。
                        最后汇总它们的报告。"""
    )

    task = "请详细调研 'LangChain DeepAgents' 框架。我需要官方的技术架构说明（来自文档）以及社区的最佳实践案例。请对比两者。"

    console.print(f"\n[bold green]用户任务:[/bold green] {task}\n")

    step = 0

    console.print("[dim]开始流式输出...[/dim]")

    try:
        async for event in agent.astream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": task
                        }
                    ]
                }
        ):
            step += 1

            for node_name, node_data in event.items():
                if node_data is None:
                    continue
                if "messages" in node_data:
                    msgs = node_data["messages"]

                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        # 过滤非消息对象
                        if not isinstance(msg, BaseMessage):
                            continue
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            tree = Tree(
                                f"[bold yellow]Step {step}: 决策与调用 (Node: {node_name})[/bold yellow]")
                            for tc in msg.tool_calls:
                                tool_name = tc['name']
                                tool_args = tc['args']

                                if tool_name == "task":
                                    # 验证成功！
                                    branch = tree.add(
                                        f"[bold red]🚀 触发 'task' 工具 (Sub-Agent)[/bold red]")
                                    branch.add(
                                        f"[cyan]子 Agent 类型:[/cyan] {tool_args.get('subagent_type')}")
                                    branch.add(
                                        f"[cyan]任务指令:[/cyan] {tool_args.get('description')}")
                                else:
                                    tree.add(f"[blue]普通工具调用:[/blue] {tool_name}")

                            console.print(tree)

                        elif isinstance(msg, ToolMessage):
                            if msg.name == "task":
                                if msg.name == "task":
                                    # Sub-Agent 完成任务返回
                                    panel = Panel(
                                        msg.content,
                                        title=f"[bold magenta]Sub-Agent 完成任务 (Node: {node_name})[/bold magenta]",
                                        border_style="magenta"
                                    )
                                    console.print(panel)
                                else:
                                    console.print(
                                        f"[dim]Tool Output ({msg.name}): {msg.content[:100]}...[/dim]")
                        elif msg.content and not msg.tool_calls:
                            title = f"[bold green]Agent 回复 (Node: {node_name})[/bold green]"
                            console.print(Panel(msg.content, title=title, border_style="green"))

    except Exception as e:
        console.print(Panel(f"[bold red]发生错误:[/bold red] {str(e)}", border_style="red"))
    console.print("\n[bold magenta]演示结束。[/bold magenta]")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_auto_subagent_demo())






