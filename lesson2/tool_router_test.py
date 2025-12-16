from model_factory import get_model
from langchain.tools import tool
from dotenv import load_dotenv
from langchain.agents import create_agent
import os
load_dotenv()

@tool
def search_web(query: str) -> str:
    """Web 搜索工具，用于查询网络公开信息，不适用于内部数据.参数：query 用户查询，如 OpenAI发布会"""
    return f"模拟搜索结果：你搜索了 {query}"

@tool
def extract_pdf_text(path: str) -> str:
    """解析 PDF 文本文件。参数为文件的本地路径.参数：path 文件路径，如 /files/contract.pdf"""
    return f"模拟 PDF 内容：从 {path} 中解析出的内容"

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询，仅限内部业务数据库.参数：sql Sql语句，如 select * from users limit 5"""
    return f"模拟 SQL 执行：{sql}"

@tool
def calculate(expr: str) -> str:
    """计算数学表达式。适用于算式运算.参数：expr 数学表达式，如 (12+3)*(8-2)"""
    return str(eval(expr))


model = get_model("qwen2.5:7b", "ollama")

# 1.Tool 工具分组
TOOL_GROUPS = {
    "search": [search_web],
    "pdf": [extract_pdf_text],
    "database": [query_database],
    "math": [calculate],
}

# 2.创建一个意图识别模型
agent = get_model("qwen2.5:7b", "ollama")


# 3. 定义意图分类系统提示
INTENT_SYSTEM_PROMPT = """
你是一个专业的意图分类器，请只返回以下类别之一：
- search
- pdf
- database
- math
- none

并严格只返回类别名，不要输出其它内容。
"""
# 4. 定义意图分类函数
def classify_intent(user_query: str) -> str:
    result = model.invoke(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            ("user", user_query)
        ]
    )
    return result.content.strip()

# print(classify_intent("总结/user/home/1.pdf 文件内容"))

# 根据用户输入创建 agent
def create_tool_agent(group_name: str):
    tools = TOOL_GROUPS.get(group_name, [])

    if not tools:
        return None

    agent = create_agent(
        model,
        tools = tools,
        system_prompt="你是一个 helpful assistant，可以使用工具回答问题。你必须严格根据工具描述选择工具！如果没有合适的工具，请回答“无合适工具”"
    )

    return agent

# 路由智能体函数
def router_agent(user_query: str):
    intent = classify_intent(user_query)
    print(f"分类意图: {intent}")

    tool_agent = create_tool_agent(intent)

    if tool_agent is None:
        return "无合适工具"

    response = tool_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": user_query}
            ]
        }
    )

    return response

res = router_agent("帮我查询型号为Kb024的耳机的库存")
print("Agent Response:", res)

queries = [
        "请帮我搜索一下今年Google最新的大模型版本的发布会",
        "帮我解析一下这个PDF：/root/files/contract.pdf",
        "执行一个SQL：select * from products limit 5",
        "计算 (17+3)*(8-1)",
    ]

for q in queries:
    print("\n====== 用户问题 ======")
    print(q)
    print("====== Agent 回复 ======")
    print(router_agent(q)["messages"][2])


