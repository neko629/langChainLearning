#%%
import os
import uuid
from typing import List

# --- 1. 导入组件 ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma # 向量数据库
from langchain_core.documents import Document
from langchain.agents import AgentState, create_agent
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from model_factory import get_model

# 确保配置了 OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = "sk-..."

# ==========================================
# 2. 初始化向量数据库 (长期记忆的物理载体)
# ==========================================
# 在生产环境中，这里应该是连接到 Pinecone, Milvus 或本地持久化的 Chroma
embeddings = OllamaEmbeddings(
    model="bge-m3:latest",
)
model = get_model("qwen2.5:7b", "ollama")

# 初始化 Chroma 向量数据库
vector_store = Chroma(
    collection_name="agent_long_term_memory",
    embedding_function=embeddings,
    #persist_directory="./chroma_db" # 如果想存到硬盘，取消注释这一行
)

# ==========================================
# 3. 定义记忆工具 (Agent 的手)
# ==========================================
# 3.1 定义记忆保存工具
# ==========================================
@tool
def save_memory(content: str):
    """
    将重要信息保存到长期记忆中。
    当你获知用户的喜好、职业、计划或其他长期有效的事实时，调用此工具。
    参数:
        content (str): 要保存的记忆内容。
    """
    print(f"\n[记忆操作] 正在保存记忆: '{content}'")
    # 将文本封装为 Document
    doc = Document(
        page_content=content,
        metadata={"source": "user_interaction", "timestamp": "simulated_time"} # 可选元数据
    )
    # 写入向量库
    vector_store.add_documents([doc])
    return "记忆已成功保存。"

# 3.2 定义记忆搜索工具
# ==========================================
@tool
def search_memory(query: str):
    """
    从长期记忆中搜索相关信息。
    当你被问及关于用户过去的问题，或者你不确定答案时，使用此工具进行查找。
    参数:
        query (str): 要搜索的查询语句。
    """
    print(f"\n[记忆操作] 正在搜索记忆: '{query}'")

    # 执行语义搜索 (k=2 表示只取最相关的2条)
    results = vector_store.similarity_search(query, k=2)


    if not results:
        return "没有找到相关的记忆。"

    # 将搜索结果拼接成字符串返回给 Agent
    memory_content = "\n".join([f"- {doc.page_content}" for doc in results])
    return f"找到以下相关记忆:\n{memory_content}"

# 将工具放入列表
tools = [save_memory, search_memory]

# ==========================================
# 4. 创建 Agent
# ==========================================

# 定义系统提示词：教会 Agent 何时使用记忆工具
SYSTEM_PROMPT = """你是一个拥有长期记忆的私人助手。
你的目标是记住用户的喜好和重要信息，以便提供个性化服务。

1. 如果用户告诉你任何关于他们自己的事实（如名字、喜好、居住地），请务必调用 'save_memory' 工具保存。
2. 如果用户问你一个问题，而答案可能在你之前的记忆中，请先调用 'search_memory' 工具查找。
3. 如果只是闲聊，不需要调用工具。
"""


# 使用 checkpointer 依然是必要的，用于维持当前这一轮对话的上下文
checkpointer = MemorySaver()

# 创建 Agent 应用
agent_app = create_agent(
    model,
    tools,
    system_prompt=SYSTEM_PROMPT, # 注入系统提示词
    checkpointer=checkpointer
)

# ==========================================
# 5. 运行演示
# ==========================================

def run_demo():
    # === 场景 A：存入记忆 ===
    # 使用一个 thread_id，代表这是今天的对话
    config_a = {"configurable": {"thread_id": "session_today"}}

    print("--- 场景 A：用户告诉 Agent 喜好 ---")
    user_input_1 = "你好，记住我最喜欢的水果是草莓，而且我对花生过敏。"

    # 运行 Agent，stream_mode="values"参数，返回每个时间步的中间结果
    for chunk in agent_app.stream({"messages": [HumanMessage(content=user_input_1)]}, config=config_a, stream_mode="values"):
        # 只打印最后一条机器人的回复
        pass
    print(f"Agent: {chunk['messages'][-1].content}")

    # === 场景 B：模拟遗忘 (开启新线程) ===
    # 我们换一个 thread_id，这意味着 Agent 失去了“短期记忆” (MemorySaver 里的东西访问不到了)
    # 但是，长期记忆在 VectorStore 里，是可以跨 thread 访问的！
    config_b = {"configurable": {"thread_id": "session_tomorrow"}}

    print("\n--- 场景 B：第二天 (新的 Session，短期记忆已清空) ---")
    user_input_2 = "我想吃点零食，但我忘了我有什么忌口，你能帮我查查吗？"

    print(f"User: {user_input_2}")

    # 观察控制台输出，你会看到 Agent 自动调用 search_memory
    final_response = None
    for chunk in agent_app.stream({"messages": [HumanMessage(content=user_input_2)]}, config=config_b, stream_mode="values"):
        final_response = chunk['messages'][-1]

    print(f"Agent: {final_response.content}")

if __name__ == "__main__":
    run_demo()