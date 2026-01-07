from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel
from openrouter_test import model as openrouter_model


class ProfileModel(BaseModel):
    name: str
    age: int
    email: str

class ProfilesModel(BaseModel):
    profiles: list[ProfileModel]

agent = create_agent(
    model = openrouter_model,
    tools = [],
    response_format = ToolStrategy(ProfilesModel)
)

query = {
    "messages": [
        {
            "role": "system",
            "content": "你是一个用户信息收集助手。请根据用户提供的信息，生成包含姓名、年龄和电子邮件的用户资料。"
        },
        {
            "role": "user",
            "content": "姓名是张伟，年龄是28岁，电子邮件是2222@qq.com, 他的同事李丽，年龄32岁，电子邮件是2fasdf@dasdf.com"
        }
    ]
}

result = agent.invoke(query)

print("Full Response:", result)

