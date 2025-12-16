from langchain.agents import create_agent
from model_factory import get_model

model = get_model("qwen2.5:7b", "ollama")
#model = get_model("gemini-2.5-flash", "google_genai")

# 测试模型
# result = model.invoke("你是谁?")
# print("Model Response:", result)
agent = create_agent(
    model = model, # 使用之前创建的模型
    tools = [], # 这里可以添加工具列表
    system_prompt = "你是一个订单查询助手, 可以查询订单的状态和明细。",
    middlewares = [], # 这里可以添加中间件列表
    checkpointer = checkpointer, # 这里可以添加检查点
    store = store, # 状态存储长期记忆,
    state_schema = OrderQueryState, # 扩展状态,
    context_schema = AgentContext, # 扩展上下文
    response_format = ResponseModel # 结构化输出
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "帮我查询订单12345的状态。"}
        ]
    },
    config = config
)