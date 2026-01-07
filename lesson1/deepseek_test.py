from langchain_deepseek import ChatDeepSeek
import os
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

# load environment variables from a .env file if it exists
from dotenv import load_dotenv

load_dotenv()

model = ChatDeepSeek(
    model = "deepseek-chat",
    temperature = 0.7,
    timeout = 30
)
#
# question = "你是谁?"
#
# # result = model.invoke(question)
# #
# # print(result)
#
# system_message = SystemMessage("你是一个的AI助手。")
# human_message = HumanMessage(question)
#
# model2 = init_chat_model(
#     "deepseek-chat",
#     temperature = 0.7,
#     timeout = 30,
#     model_provider = "deepseek"
# )
# result2 = model2.invoke([system_message, human_message])
# print(result2.content_blocks)

