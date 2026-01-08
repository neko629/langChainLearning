from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from openrouter_test import model as gpt_model

summarization_mw = SummarizationMiddleware(
    model = gpt_model,
    trigger = ("tokens", 100),
    keep = ("messages", 1)
)

checkpointer = InMemorySaver()

agent = create_agent(
    model = gpt_model,
    middleware=[summarization_mw],
    checkpointer = checkpointer
)

queries = (
    "1 + 1 等于多少？",
    "地球到月球的距离是多少？",
    "2 + 2 等于多少？",
    "太阳系中最大的行星是哪一个？",
    "3 + 3 等于多少？",
    "中国的首都在哪"
)

for i, query in enumerate(queries):
    response = agent.invoke(
        {
            "messages": [{"role": "user", "content": query}]
        },
        config = {"configurable": {"thread_id": "1"}}
    )
    print(f"\nResponse to query {i+1}:", response)