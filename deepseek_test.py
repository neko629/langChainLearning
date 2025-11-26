from langchain_deepseek import ChatDeepSeek
import os
from langchain.chat_models import init_chat_model

# load environment variables from a .env file if it exists
from dotenv import load_dotenv
load_dotenv()

model = ChatDeepSeek(
    model = "deepseek-chat",
    temperature = 0.7,
    timeout = 30
)

question = "你是谁?"

# result = model.invoke(question)
#
# print(result)

model2 = init_chat_model(
    "deepseek-chat",
    temperature = 0.7,
    timeout = 30,
    model_provider = "deepseek"
)
result2 = model2.invoke(question)
print(result2)

