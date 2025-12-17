from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

def get_model(model_name: str, model_provider: str, temperature: float = 0.7, timeout: int = 30):
    model = init_chat_model(
        model_name,
        temperature = temperature,
        timeout = timeout,
        model_provider = model_provider
    )
    return model