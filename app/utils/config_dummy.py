from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    
    ## The remote ollama url
    # ollama_base_url: str = ""

    ## In local venv use the localhost url or 0.0.0.0 for ollama otherwise will through Error;
    ## Ollama not available, chat and vission model will be None
    ollama_base_url: str = "http://localhost:11434" #"http://0.0.0.0:11434" #this is only for local #"http://ollama:11434" # This is for docker 
    ollama_model: str = "mistral-nemo:12b" #"qwen3:30b" #"deepseek-r1:32b" # qwen3:30b "llama3.3:70b" # "llama4"  "llama3.3:70b" nemotron-mini:latest
    ollama_vision_model: str = "llava:latest" #llama3.2-vision:11b" H100 has issues with this vision model
    redis_url: str = "redis://:" #"red"
    
    ## Change the embedding model provider and embedding model at the same time,
    ## Use huggingface embedding models only when the model provider is huggingface...
    embedding_provider: str = "ollama" #"huggingface" or  #ollama
    embedding_model: str = "mxbai-embed-large:latest" #"sentence-transformers/all-mpnet-base-v2" #mxbai-embed-large:latest 669MBs; nomic-embed-text:latest 274MBs
    
    temp_dir: str = "./temp"
    vector_store_dir: str = "./vector_stores"
    llm_temperature: float = 0.3
  
    
    ollama_timeout: int = 1200  # 20 minutes
    model_pull_timeout: int = 1800  # 30 minutes for large models
    min_model_size: int = 100_000_000  # 100MB :TODO Implment this check while pulling the models


    ## SQL Agent variables
    default_database_url: str = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'app', 'Chinook.db')}" #"sqlite:///test.db"
    sql_agent_temperature: float = 0.1
    sql_agent_max_tokens: int = 4096

    ## Openai API Key for Web_search
    openai_api_key: str ="sk-proj--UX-hu3phhAA"
    web_search_rate_limit: int = 10
    web_search_rate_window: int = 60



    ## Suadeo Search Config - Used in the model schemas
    min_similarity_threshold: float = 0.7
    min_context_chunks: int = 2
    index_rebuild_threshold: int = 1000  # Rebuild index if data changes exceed this

    class Config:
        env_file = ".env"

settings = Settings()
