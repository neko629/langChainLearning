from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(
    "gemini-2.5-flash",
    temperature=0.7,
    timeout=30,
    model_provider="google_genai"
)
question = "用一段话描述大海"
full = ""
for chunk in model.stream(question):
    if chunk.content:
        full += chunk.content
        print(full)