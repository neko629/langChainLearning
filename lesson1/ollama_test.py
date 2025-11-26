from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_core.rate_limiters import InMemoryRateLimiter


rate_limiter = InMemoryRateLimiter(
    requests_per_second = 1,
    check_every_n_seconds = 1
)
load_dotenv()

model = init_chat_model(
    "qwen2.5:7b",
    temperature=0.7,
    timeout=30,
    model_provider="ollama",
    rate_limiter = rate_limiter
)

model = model.with_retry(
    stop_after_attempt = 3,
    wait_exponential_jitter = True
)
question = "你是谁?"

templeate = PromptTemplate(
    input_variables=["question"],
    template="请用{num}句话以{style}的方式回答以下问题：{question}",
    partial_variables={"style": "悲伤", "num": 5}
)

system_message = SystemMessage("你是一个的AI助手。")
human_message = HumanMessage(templeate.partial(style="幽默", num=1).format(question=question))
ai_message = AIMessage("请问要用什么语言回答?")
user_message = HumanMessage("日语。")

chat_message = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个的AI助手。"),
        ("human", "请用{language}回答以下问题：你是谁?"),
    ]
)


#ans = model.invoke([system_message, human_message, ai_message, user_message])
ans = model.invoke(chat_message.format_messages(
    language="中文"
))
print(ans)
# for i in range(10):
#     result = model.invoke(question)
#     print(f"Response {i+1}: {result}")

# embedding = init_embeddings(
#     "bge-m3",
#     provider="ollama"
# )
# res = embedding.embed_query(question)
# print(res)