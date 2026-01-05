import json
from rich.json import JSON
from rich.console import Console
from rich.panel import Panel
from deepagents_demo import agent

console = Console()


def debug_agent(query: str, save_to_file: str = None):
    console.print(Panel.fit(
        f"[bold cyan]查询:[/bold cyan] {query}",
        border_style="cyan"
    ))

    step_num = 0
    final_response = None

    config = {"configurable": {"thread_id": "2"}}

    for event in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        stream_mode = "values",
        config = config
    ):
        step_num += 1
        console.print(f"\n[bold yellow]{'─' * 80}[/bold yellow]")
        console.print(f"[bold yellow]步骤 {step_num}[/bold yellow]")
        console.print(f"[bold yellow]{'─' * 80}[/bold yellow]")

        if "messages" in event:
            messages = event["messages"]
            if messages:
                msg = messages[-1]
                if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                    # 没有工具调用, 是最终响应
                    final_response = msg.content
                if hasattr(msg, 'content') and msg.content:
                    # AI 思考
                    content = msg.content
                    if len(content) > 300 and not (hasattr(content, 'tool_calls') and msg.tool_calls):
                        preview = content[:300] + "..."
                        console.print(Panel(
                            f"{preview}\n\n[dim](内容较长,完整内容将在最后显示)[/dim]",
                            title="[bold green]AI 思考[/bold green]",
                            border_style="green"
                        ))
                    else:
                        console.print(Panel(
                            content,
                            title="[bold green]AI 思考[/bold green]",
                            border_style="green"
                        ))
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # 工具调用
                    for tool_call in msg.tool_calls:
                        tool_info = {
                            "工具名称": tool_call.get("name", "不明工具"),
                            "参数": tool_call.get("args", {})
                        }
                        console.print(Panel(
                            JSON(json.dumps(tool_info, ensure_ascii=False)),
                            title="[bold blue]工具调用[/bold blue]",
                            border_style="blue"
                        ))
                # 工具响应
                if hasattr(msg, 'name') and msg.name:
                    response = str(msg.content)[:500]
                    if len(str(msg.content)) > 500:
                        response += f"\n..(共{len(str(msg.content))}个字符)"
                    console.print(Panel(
                        response,
                        title=f"[bold magenta]工具响应: {msg.name}[/bold magenta]",
                        border_style="magenta"
                    ))

    console.print("\n[bold green]任务完成![/bold green]\n")
    return final_response
print("调试方法已创建")

query = "详细调研 LangChain DeepAgents 框架的核心特性，并写一份结构化的总结报告。"

# 使用调试函数）
result = debug_agent(query)



