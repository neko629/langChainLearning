from langchain_core.messages import BaseMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from model_factory import get_model
from deepagents import create_deep_agent
from langchain_community.tools import TavilySearchResults

console = Console()

async def run_auto_subagent_demo():
    console.print(
        Panel.fit("[bold magenta]DeepAgents 自动 SubAgent 中间件演示[/bold magenta]",
                  border_style="magenta"))
    console.print(
        "[dim]本演示验证：即使不传入 subagents 参数，Agent 默认也会启用 'general-purpose' 子 Agent。[/dim]")

    deepseek_model = get_model(
        'deepseek-chat', 'deepseek', temperature=0, timeout=300)

    tavily_tool = TavilySearchResults(max_results=2)

    tools = [tavily_tool]

    console.print("[bold cyan]正在创建 Agent (subagents=None)...[/bold cyan]")

    agent = create_deep_agent(
        model = deepseek_model,
        tools = tools,
        system_prompt = """你是一个能够高效处理并发任务的智能助手。
                        对于包含多个独立部分的复杂任务，你必须使用 'task' 工具来创建 'general-purpose' 子 Agent 进行处理。
                        不要自己在主线程中串行执行所有操作。利用子 Agent 来隔离上下文并提高效率。"""
    )

    task = """请同时调研以下两个完全不同的主题，并分别给出简短总结：
                1. Python 语言的历史起源。
                2. Rust 语言的内存安全机制。
                请务必使用子 Agent 分别处理这两个任务。"""

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






