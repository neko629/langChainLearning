from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from openai import timeout

load_dotenv()

model = init_chat_model(
    "gemini-2.5-flash",
    temperature=0.7,
    timeout = 3,
    model_provider="google_genai"
)
question = "你是谁?"
result = model.invoke(question)
print(result)