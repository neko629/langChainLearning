from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from model_factory import get_model

model = get_model('deepseek-chat', 'deepseek', temperature=0.7, timeout=30)

backend = FilesystemBackend(root_dir = "./agent_data", virtual_mode = True)

agent = create_deep_agent(
    model,
    backend = backend,
    system_prompt = "你是一个文件系统操作助手。请根据用户指令使用相应的工具。"
)

def run_test(task_name, instruction):
    print(f"\n🔹 [测试: {task_name}]")
    print(f"指令: {instruction}")
    try:
        # 使用 invoke 而不是 stream 以简化输出
        result = agent.invoke({"messages": [("user", instruction)]})
        last_msg = result["messages"][-1]
        print(f"🤖 Agent 回复: {last_msg.content}")

        # 打印工具调用详情 (如果有)
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool in msg.tool_calls:
                    print(f"🛠️  调用工具: {tool['name']} args={tool['args']}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")\

# run_test("write_file",
#          "请在当前目录下创建一个名为 'hello_world.py' 的文件，内容是：\nprint('Hello from DeepAgents!')")
#
# run_test("ls", "请列出当前目录下的所有文件，确认 hello_world.py 是否存在。")

#run_test("read_file", "请读取 'hello_world.py' 的内容并展示给我。")

#run_test("edit_file",
#         "请修改 'hello_world.py' 文件，将 print 内容改为 'Hello from Modified File!'。")
#run_test("grep", "请在当前目录下搜索包含 'Modified' 字符串的文件。")

#run_test("glob", "请找出当前目录下所有的 .py 文件。")

print("\n🔹 [测试: execute]")
print("指令: 尝试运行 hello_world.py 脚本")
try:
    # 我们尝试强行要求 Agent 运行，看它如何反应
    # 如果没有 execute 工具，Agent 可能会说无法执行
    response = agent.invoke({"messages": [("user", "请使用 execute 工具运行 python hello_world.py")]})
    print(f"🤖 Agent 回复: {response['messages'][-1].content}")
except Exception as e:
    print(f"⚠️ 测试说明: execute 工具可能不可用 (取决于 Backend 支持): {e}")

print("\n" + "="*50)
print("✅ 测试结束")
