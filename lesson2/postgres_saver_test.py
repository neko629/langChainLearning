from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from model_factory import get_model

print("\n" + "=" * 60)
print("场景 2: Postgres 持久化记忆（生产环境）")
print("=" * 60)

@tool
def get_user_info(name: str) -> str:
    """查询用户信息，返回姓名、年龄、爱好"""
    user_db = {
        "陈明": {"age": 28, "hobby": "旅游、滑雪、喝茶"},
        "张三": {"age": 32, "hobby": "编程、阅读、电影"}
    }
    info = user_db.get(name, {"age": "未知", "hobby": "未知"})
    return f"姓名: {name}, 年龄: {info['age']}岁, 爱好: {info['hobby']}"

model = get_model("qwen2.5:7b", "ollama")

DB_URL = "postgresql://postgres:123456@localhost:5432/langchain_db"

with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    # 自动创建表结构（仅首次运行需要）
    checkpointer.setup()

    agent = create_agent(
        model,
        tools = [get_user_info],
        checkpointer = checkpointer
    )

    # 配置线程 ID（用于区分不同用户）
    config = {"configurable": {"thread_id": "production_user_001"}}

    # 模拟用户注册流程
    agent.invoke(
        {"messages": [{"role": "user", "content": "我是新用户张三，请记录我的信息"}]},
        config=config
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "我是谁？"}]},
        config=config
    )
    print(f"AI: {response['messages'][-1].content}")