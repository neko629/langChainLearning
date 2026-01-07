from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, chain

from openrouter_test import model as openrouter_model

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个的AI翻译助手, 会以 JSON 格式返回回答内容。"),
        ("human", "请将以下内容翻译成{language}：\n{content}"),
    ]
)
my_parser = JsonOutputParser()

# 自定义链
# 1. 使用@chain
# 2. RunnableLambda 包装

@chain
def get_language(language: str):
    return language

@chain
def _sentencte(input: None):
    return "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、解决问题、理解自然语言和感知环境。人工智能技术广泛应用于各个领域，如医疗、金融、交通和娱乐，推动了自动化和创新的发展。"

def get_content(content: str):
    return content


chain1 = prompt_template | openrouter_model
chain2 = (
    {
        "language": RunnableLambda(lambda x: "英文")| get_language,
        "content": _sentencte | RunnableLambda(get_content),
    } | chain1 | my_parser
)



response = chain2.invoke({})

print(response)
