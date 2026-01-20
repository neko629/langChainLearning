from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_community.callbacks.uptrain_callback import handler
from langgraph.checkpoint.memory import InMemorySaver

from openrouter_test import model as openrouter_model
from lesson1.deepseek_test import model as deepseek_model


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])
    if message_count > 3:
        print(f"消息数量为 {message_count}, 使用 DeepSeek 模型。")
        model = deepseek_model
    else:
        print(f"消息数量为 {message_count}, 使用 OpenRouter 模型。")
        model = openrouter_model

    return handler(request.override(model = model))


checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model = openrouter_model,
    middleware = [dynamic_model_selection],
    checkpointer = checkpointer,
)

for i in range(5):
    question = "1 + 2 等于多少?"  # 简单问题
    response = agent.invoke(
        {
            "messages": [{"role": "user", "content": question}]
        },
        config = config,
    )
    print(f"Response to question {i+1}:", response)


