import asyncio
from app.services.llm_service import llm_service
from app.services.model_manager import model_manager
from app.utils.logging import get_logger
from app.utils.config import settings

logger = get_logger("app.preload")

async def preload_models():
    """Ensure high-priority models are preloaded before the application starts accepting requests"""
    logger.info("Preloading models...")
    
    # Check Ollama availability
    if llm_service.check_and_init_models():
        # Register models with the model manager
        # High priority (2) for the main chat model that should always stay loaded
        model_manager.register_model(settings.ollama_model, preload=True, priority=1)
        
        # Medium priority (1) for the embedding model
        model_manager.register_model(settings.embedding_model, preload=True, priority=2)
        
        # Lower priority (0) for models that should be loaded on demand
        model_manager.register_model(settings.ollama_vision_model, preload=False, priority=0)
        
        # These calls will already pull/load models if needed
        embeddings = llm_service.get_embeddings()
        chat_model = llm_service.get_chat_model()
        
        logger.info("All priority models successfully preloaded")
        return True
    else:
        logger.warning("Ollama service not available, models could not be preloaded")
        return False