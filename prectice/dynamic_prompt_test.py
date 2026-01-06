from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from openrouter_test import model

class UserRoleContext(TypedDict):
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.get("user_role", "user")
    if user_role == "expert":
        return "请用专业术语回答问题。"
    elif user_role == "bigineer":
        return "请提供简洁和易懂的回答, 不要使用复杂的术语。"
    else:
        return "请三句话之内回答问题, 并在最后推荐用户关注本平台。"

agent = create_agent(
    model = model,
    middleware = [user_role_prompt],
    context_schema = UserRoleContext,
)

question = "什么是机器学习?"

response_expert = agent.invoke(
    {
        "messages": [{"role": "user", "content": question}]
    },
    context={"user_role": "expert"}
)
print("Expert Response:", response_expert)

response_bigineer = agent.invoke(
    {
        "messages": [{"role": "user", "content": question}]
    },
    context={"user_role": "bigineer"}
)
print("Bigineer Response:", response_bigineer)

response_default = agent.invoke(
    {
        "messages": [{"role": "user", "content": question}]
    },
    context={"user_role": "other"}
)
print("Default Response:", response_default)


