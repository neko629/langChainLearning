from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model(
    "qwen2.5:7b",
    temperature=0.7,
    timeout=30,
    model_provider="ollama",
)

system_message = SystemMessage(
    content = "你叫超级聪明小智，是一个乐于助人的AI助手。"
)

messages = [system_message] # 初始化消息列表
print("* 输入 qqq 退出对话\n")
print("小智: 请先告诉我你的名字吧！")
user_name = input("我的名字是: ")
print("小智: 很高兴认识你，" + user_name + "！有什么我可以帮你的吗？")

while True:
    user_input = input(f"{user_name}: ")
    if user_input.lower() == "qqq":
        print("小智: 再见！期待下次和你聊天！")
        break
    human_message = HumanMessage(content=user_input)
    messages.append(human_message)

    # 实时输出模型生成内容
    print("小智: ", end="", flush=True) # 不换行输出
    full_response = ""
    for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end = "", flush=True) # 不换行输出
            full_response += chunk.content
    print("\n" + "~" * 10) # 换行
    ai_message = AIMessage(content=full_response)
    messages.append(ai_message)
    messages = messages[-10:]  # 保持消息列表的长度，避免过长
