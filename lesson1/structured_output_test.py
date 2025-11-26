from typing import List
from langchain_core.utils.pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
import json

load_dotenv()


class Person(BaseModel):
    # Define the structure of the Person object
    name: str = Field(..., description="The person's full name")
    age: int = Field(..., description="The person's age in years", ge=10, le=100)
    high: float = Field(..., description="The person's height in meters")
    hobbies: List[str] = Field(..., description="A list of the person's hobbies")


llm = ChatOpenAI(
    model="gpt-4o-mini"
)

structured_llm = llm.with_structured_output(Person, include_raw=True)

json_parser = JsonOutputParser(pydantic_object=Person)

prompt = """提取名为{full_name}的人的信息, 他今年99岁, 身高1.75米, 爱好是篮球, 音乐和旅行.
请以JSON格式返回, 包含以下字段:
- name: 全名
- age: 年龄
- high: 身高(米)
- hobbies: 爱好列表

"""
#
# result = structured_llm.invoke(prompt)
# print(result)
# print(type(result))
# print(isinstance(result, Person))

model = init_chat_model(
    "qwen2.5:7b",
    temperature=0.0,
    timeout=30,
    model_provider="ollama",
)

agent = create_agent(
    model=model,
    tools=[],
    response_format=Person
)
# result2 = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     }
# )
# print(result2)
# print("----------------")
# print(result2["structured_response"])

prompt2 = ChatPromptTemplate.from_template(prompt)

runnable = prompt2 | model | json_parser
result3 = runnable.invoke(
    {"full_name": "张伟"}
)
print(result3)
