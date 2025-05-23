import numpy as np
import requests
import logging
from typing import List
from app.exceptions.suadeo_exceptions import OllamaConnectionError, EmbeddingGenerationError

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.suadeo_embedding_services")

class OllamaEmbeddings:
    """
    Ollama embeddings service for RAG system
    """
    
    def __init__(self, model: str = "mxbai-embed-large:latest", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama embeddings service
        
        Args:
            model: Embedding model to use
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embeddings"
        
        # Test connection on initialization
        self._test_connection()
        logger.info(f"Successfully connected to Ollama at {base_url}")
    
    def _test_connection(self):
        """Test if Ollama is running and model is available"""
        test_payload = {
            "model": self.model,
            "prompt": "test"
        }
        
        try:
            response = requests.post(self.api_url, json=test_payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of embeddings
        """
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            embedding = np.array(result["embedding"], dtype=np.float32)
            return embedding
            
        except Exception as e:
            raise EmbeddingGenerationError(f"Error generating embedding: {e}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 40) -> np.ndarray:
        """
        Get embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of all embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_embeddings = []
            for text in batch:
                embedding = self.embed_text(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)