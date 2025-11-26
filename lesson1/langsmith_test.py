from langsmith import Client
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
load_dotenv()

api_key = os.getenv("LANGSMITH_API_KEY")

client = Client(api_key=api_key)
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
print(prompt)

formated = prompt.format(context="三三是一名中学生, 今天19岁", question="谁是三三?")

print(formated)

model = init_chat_model(
    "qwen2.5:7b",
    temperature=0.7,
    timeout=30,
    model_provider="ollama"
)

result = model.invoke(formated)
print(result.content_blocks)