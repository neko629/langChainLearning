import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")

model = init_chat_model(
    model="openai/gpt-4o-mini",
    model_provider="openai",
    base_url=openrouter_base_url,
    api_key=openrouter_api_key
)

# query = "10句话介绍一下人工智能的发展历史。"
#
# response = model.invoke(query)
#
# print(response.content)

