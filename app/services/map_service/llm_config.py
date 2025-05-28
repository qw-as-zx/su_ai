from langchain_ollama import ChatOllama
from app.utils.config import settings


base_url = settings.ollama_base_url
model =  settings.ollama_model
llm = ChatOllama(model=model, base_url=base_url, temperature=0.01)
