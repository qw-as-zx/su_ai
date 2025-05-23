import requests
import subprocess
from typing import Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.utils.config import settings
import ollama
from typing import Dict, List, Optional

from app.exceptions.custom_exceptions import LLMServiceError
from openai import AsyncOpenAI, OpenAIError
from app.utils.redis_client import redis_client
import time
import base64
import asyncio
from app.utils.logging import get_logger
import os

# Create a logger specific to this module
logger = get_logger("app.services.llm_services")

class LLMService:
    def __init__(self):
        logger.info("LLMService initializing...Checking for Ollama & the required Models")
        self.ollama_available = self._check_ollama_available()
        self.embeddings = self._init_embeddings()
        self.chat_model = self._init_chat_model()
        self.vision_model = self._init_vision_model()


    def _check_ollama_available(self) -> bool:
        """Check if Ollama server is running and reachable"""
        try:
            response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            logger.warning("Ollama server not available, falling back to HuggingFace")
            return False

    def _ensure_model_pulled(self, model_name: str) -> bool:
        """Ensure the specified model is pulled in Ollama using the API"""
        try:
            # Check if model exists locally
            response = requests.post(
                f"{settings.ollama_base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Model {model_name} already exists locally")
                return True
            
            # If not, pull the model using the API
            logger.info(f"Pulling model {model_name}...")
            response = requests.post(
                f"{settings.ollama_base_url}/api/pull",
                json={"name": model_name},
                timeout=3000  # Pulling can take time
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to pull model {model_name}: {response.text}")
                return False
                
            logger.info(f"Successfully pulled model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking/pulling model {model_name}: {e}")
            return False
        
    def _init_embeddings(self):
        """Initialize embeddings with fallback logic"""
        try:
            if settings.embedding_provider == "ollama" and self.ollama_available:
                if not self._ensure_model_pulled(settings.embedding_model):
                    raise LLMServiceError(f"Failed to pull embedding model {settings.embedding_model}")
                
                return OllamaEmbeddings(
                    model=settings.embedding_model,
                    base_url=settings.ollama_base_url,
                    temperature=0.1,
                    keep_alive=1200
                )
            else:
                logger.info(f"Using HuggingFace embeddings with model: {settings.embedding_model}")
                return HuggingFaceEmbeddings(
                    model_name=settings.embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            logger.error(f"Embeddings initialization failed: {e}")
            raise LLMServiceError(f"Could not initialize embeddings: {e}")

    def _init_chat_model(self) -> Optional[ChatOllama]:
        """Initialize chat model with fallback logic"""
        try:
            if not self.ollama_available:
                logger.warning("Ollama not available, chat model will be None")
                return None
                
            if not self._ensure_model_pulled(settings.ollama_model):
                raise LLMServiceError(f"Failed to pull chat model {settings.ollama_model}")
            
            return ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=settings.llm_temperature,
                streaming=True,
                top_k=10,
                top_p=0.9,
                repeat_penalty=1.1,  # Fixed typo in repeat_penalty
                stop=["<|endoftext|>", "<|im_end|>"],
                keep_alive=1200
            )
        except Exception as e:
            logger.error(f"Chat model initialization failed: {e}")
            raise LLMServiceError(f"Could not initialize chat model: {e}")

    def _init_vision_model(self) -> Optional[ChatOllama]:
        """Initialize vision model with fallback logic"""
        try:
            if not self.ollama_available:
                logger.warning("Ollama not available, Vision model will be None")
                return None
                
            if not self._ensure_model_pulled(settings.ollama_vision_model):
                raise LLMServiceError(f"Failed to pull vision model {settings.ollama_vision_model}")
            
            current_dir = os.path.dirname(__file__)
            image_path = os.path.join(current_dir, "vision_model_test.png")

            logger.info(f"Starting image analysis for {image_path}")
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Create a ChatOllama instance for the vision model instead of using ollama.chat() directly
            return ChatOllama(
                model=settings.ollama_vision_model,
                base_url=settings.ollama_base_url,  # This ensures we use the correct URL
                temperature=settings.llm_temperature,
                streaming=True,
                keep_alive=1200
            )
    

            # return ollama.chat(model=settings.ollama_vision_model,
                               
            #                    messages=[{
            #                        'role':'user',
            #                        'content':'what is in this image?',
            #                        'images':[image_data]
            #                    }],
            #                    keep_alive=1200
            #                    )
        except Exception as e:
            logger.error(f"Chat vision initialization failed: {e}")
            raise LLMServiceError(f"Could not initialize vision model: {e}")
            
            
            
    def check_and_init_models(self):
        """Check if Ollama is available and initialize models if needed"""
        # First check if Ollama is running
        self.ollama_available = self._check_ollama_available()
        
        # If Ollama became available after initialization, reinitialize components
        if self.ollama_available:
            if self.embeddings is None or (settings.embedding_provider == "ollama" and 
                                           not isinstance(self.embeddings, OllamaEmbeddings)):
                self.embeddings = self._init_embeddings()
                
            if self.chat_model is None:
                self.chat_model = self._init_chat_model()
            if self.vision_model is None:
               self.vision_model = self._init_vision_model()
        return self.ollama_available
    
    def get_embeddings(self):
        """Get the current embeddings model"""
        if self.embeddings is None:
            self.ollama_available = self._check_ollama_available()
            self.embeddings = self._init_embeddings()
        return self.embeddings
    
    def get_chat_model(self):
        """Get the current chat model"""
        if self.chat_model is None and self.ollama_available:
            self.chat_model = self._init_chat_model()
        return self.chat_model

    def get_vision_model(self):
        if self.vision_model is None and self.ollama_available:
            self.vision_model = self._init_vision_model()
        return self.vision_model
    
    def is_ollama_available(self):
        """Return whether Ollama is available"""
        return self.ollama_available


# Initialize the service once as a singleton
try:
    llm_service = LLMService()
except Exception as e:
    logger.critical(f"Failed to initialize LLMService: {e}")
    raise