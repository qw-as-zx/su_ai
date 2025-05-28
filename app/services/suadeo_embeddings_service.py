# import numpy as np
# import requests
# import logging
# from typing import List
# from app.exceptions.suadeo_exceptions import OllamaConnectionError, EmbeddingGenerationError
# from app.utils.config import settings
# from app.utils.logging import get_logger

# # Create a logger specific to this module
# logger = get_logger("app.routers.suadeo_embedding_services")

# class OllamaEmbeddings:
#     """
#     Ollama embeddings service for RAG system
#     """
    
#     def __init__(self, model: str = "mxbai-embed-large:latest", base_url: str = settings.ollama_base_url):
#         """
#         Initialize Ollama embeddings service
        
#         Args:
#             model: Embedding model to use
#             base_url: Ollama server URL
#         """
#         self.model = model
#         self.base_url = base_url
#         self.api_url = f"{base_url}/api/embeddings"
        
#         # Test connection on initialization
#         self._test_connection()
#         logger.info(f"Successfully connected to Ollama at {base_url}")
    
#     def _test_connection(self):
#         """Test if Ollama is running and model is available"""
#         test_payload = {
#             "model": self.model,
#             "prompt": "test"
#         }
        
#         try:
#             response = requests.post(self.api_url, json=test_payload, timeout=10)
#             response.raise_for_status()
#         except Exception as e:
#             raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")
    
#     def embed_text(self, text: str) -> np.ndarray:
#         """
#         Get embedding for a single text
        
#         Args:
#             text: Text to embed
            
#         Returns:
#             numpy array of embeddings
#         """
#         payload = {
#             "model": self.model,
#             "prompt": text
#         }
        
#         try:
#             response = requests.post(self.api_url, json=payload, timeout=30)
#             response.raise_for_status()
            
#             result = response.json()
#             embedding = np.array(result["embedding"], dtype=np.float32)
#             return embedding
            
#         except Exception as e:
#             raise EmbeddingGenerationError(f"Error generating embedding: {e}")
    
#     def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
#         """
#         Get embeddings for multiple texts in batches
        
#         Args:
#             texts: List of texts to embed
#             batch_size: Number of texts to process at once
            
#         Returns:
#             numpy array of all embeddings
#         """
#         embeddings = []
        
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
#             logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
#             batch_embeddings = []
#             for text in batch:
#                 embedding = self.embed_text(text)
#                 batch_embeddings.append(embedding)
            
#             embeddings.extend(batch_embeddings)
        
#         return np.array(embeddings, dtype=np.float32)



import numpy as np
import requests
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.exceptions.suadeo_exceptions import OllamaConnectionError, EmbeddingGenerationError
from app.utils.config import settings
from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.suadeo_embedding_services")

class OllamaEmbeddings:
    """
    Optimized Ollama embeddings service for RAG system
    """
    
    def __init__(self, model: str = "mxbai-embed-large:latest", base_url: str = settings.ollama_base_url):
        """
        Initialize Ollama embeddings service
        
        Args:
            model: Embedding model to use
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embeddings"
        
        # Create a session for connection pooling
        self.session = requests.Session()
        # Configure session for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
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
            response = self.session.post(self.api_url, json=test_payload, timeout=10)
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
            response = self.session.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            embedding = np.array(result["embedding"], dtype=np.float32)
            return embedding
            
        except Exception as e:
            raise EmbeddingGenerationError(f"Error generating embedding: {e}")
    
    def _embed_single_text(self, text: str, index: int) -> tuple:
        """
        Helper method for embedding a single text with index tracking
        
        Args:
            text: Text to embed
            index: Original index of the text
            
        Returns:
            tuple of (index, embedding)
        """
        try:
            embedding = self.embed_text(text)
            return (index, embedding)
        except Exception as e:
            logger.error(f"Failed to embed text at index {index}: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 50, max_workers: int = 50) -> np.ndarray:
        """
        Get embeddings for multiple texts using parallel processing
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch (reduced default for better memory management)
            max_workers: Maximum number of concurrent threads
            
        Returns:
            numpy array of all embeddings
        """
        if not texts:
            return np.array([])
        
        # Pre-allocate array for better memory efficiency
        total_texts = len(texts)
        embeddings = [None] * total_texts
        
        # Process texts in batches with parallel execution
        for batch_start in range(0, total_texts, batch_size):
            batch_end = min(batch_start + batch_size, total_texts)
            batch_texts = texts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            batch_num = (batch_start // batch_size) + 1
            total_batches = ((total_texts - 1) // batch_size) + 1
            logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            # Use ThreadPoolExecutor for parallel processing within batch
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_texts))) as executor:
                # Submit all tasks for the current batch
                future_to_index = {
                    executor.submit(self._embed_single_text, text, idx): idx 
                    for text, idx in zip(batch_texts, batch_indices)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    try:
                        original_index, embedding = future.result()
                        embeddings[original_index] = embedding
                    except Exception as e:
                        logger.error(f"Failed to process embedding in batch {batch_num}: {e}")
                        raise EmbeddingGenerationError(f"Batch processing failed: {e}")
        
        # Convert to numpy array
        return np.array(embeddings, dtype=np.float32)
    
    def embed_batch_sequential(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Fallback method: Get embeddings sequentially (original implementation with session reuse)
        Use this if parallel processing causes issues with your Ollama setup
        
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
    
    def __del__(self):
        """Clean up session on object destruction"""
        if hasattr(self, 'session'):
            self.session.close()