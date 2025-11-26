import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个专业的老师"),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-4o",
)

chain = prompt | llm

events = chain.astream_events(
    {"question": "请解释一下相对论的基本原理。"},
    version = "v1"
)
async def main():
    events = chain.astream_events(
        {"question": "请解释一下相对论的基本原理。"},
        version="v1"
    )

    async for event in events:
        print(f"Event: {event}")
        if "data" in event:
            print(event["data"])

        print("-----------------------\n")

if __name__ == "__main__":
    asyncio.run(main())