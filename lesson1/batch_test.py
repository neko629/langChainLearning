import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
import asyncio
load_dotenv()

config = RunnableConfig(
    max_concurrency = 10
)

model = init_chat_model(
    "qwen2.5:7b",
    temperature=0.7,
    timeout=30,
    model_provider="ollama",
)
start_time = time.time()
result = asyncio.run(
    model.abatch(
        [
            "你支持多模态吗?",
            "你是谁?",
            "今天天气怎么样?",
            "请介绍一下你自己。",
            "你喜欢什么颜色？"
            "你会说哪些语言？",
            "你能帮我做什么？",
            "你怎么看待人工智能的发展？",
            "你喜欢旅行吗？",
            "你最喜欢的食物是什么？",
            "你有什么兴趣爱好？"
            "你觉得未来的科技会怎样？"
        ],
        config=config
    )
)
end_time = time.time()
print(f"Total time for batch: {end_time - start_time} seconds")
for res in result:
    print(res.content_blocks)